"""D12's own entry point: verify, inventory, and time the fixed costs.

`scripts/d11_diagnose.py` refuses any branch but `d11`, and D12 must not weaken that
guard to let its own work through, so D12 checks the same conditions from here instead.
Each stage is timed and each stage is allowed to fail on its own, because the point of
day 1 is to learn *everything* that is missing, not to stop at the first gap.

    python3 scripts/d12_preflight.py --scope local
    python3 scripts/d12_preflight.py --scope local --hash-inputs   # the 3.9 GB fixed cost

Nothing here writes into reports/d11 or touches a saved run.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repo import (D12Error, installed_versions, manifest_coverage, missing_packages,
                          registered_sha256, required_inputs, sha256, verify_repository)

RUNS = {"d11_v1": "reports/d11/d11_v1", "d11_diagnostic_v1": "reports/d11/d11_diagnostic_v1"}


class Report:
    """Ordered stage results, each with its own status, detail and wall clock."""

    def __init__(self):
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name):
        entry = {"stage": name, "status": "ok", "seconds": None, "detail": {}}
        self.stages.append(entry)
        started = time.perf_counter()
        try:
            yield entry
        except Exception as error:  # a missing input must not hide the next check
            entry["status"] = "failed"
            entry["detail"]["error"] = f"{type(error).__name__}: {error}"
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 3)

    @property
    def blocking(self):
        return [entry for entry in self.stages if entry["status"] != "ok"]


def check_repository(report, baseline):
    with report.stage("repository") as entry:
        entry["detail"] = verify_repository(ROOT, baseline)


def check_inputs(report, config):
    with report.stage("required_inputs") as entry:
        required = required_inputs(config)
        absent = [str(path) for path in required if not (ROOT / path).is_file()]
        provider = ROOT / Path(config["data"]["qlib_provider_uri"]).expanduser()
        provider_ready = (provider / "features").is_dir() and (provider / "calendars/day.txt").is_file()
        entry["detail"] = {"declared": len(required), "absent": len(absent),
                           "absent_examples": absent[:8], "qlib_provider": str(provider),
                           "qlib_provider_ready": provider_ready}
        if absent or not provider_ready:
            entry["status"] = "absent"


def check_registered_digests(report):
    with report.stage("registered_sha256") as entry:
        results, bad = {}, []
        for relative, expected in registered_sha256(ROOT).items():
            path = ROOT / relative
            if not path.is_file():
                results[relative], _ = "absent", bad.append(relative)
                continue
            observed = sha256(path)
            results[relative] = "ok" if observed == expected else f"mismatch:{observed[:16]}"
            if observed != expected:
                bad.append(relative)
        entry["detail"] = {"files": results, "failed": bad}
        if bad:
            entry["status"] = "mismatch"


def check_environment(report, config):
    with report.stage("environment") as entry:
        summary = json.loads((ROOT / "reports/d9_v2_summary.json").read_text())
        pinned = summary["repository"]["dependency_versions"]
        observed = installed_versions()
        wrong = {name: f"{value} != {pinned[name]}" for name, value in observed.items()
                 if value != pinned[name]}
        absent = missing_packages()
        entry["detail"] = {"python": platform.python_version(), "platform": platform.platform(),
                           "missing_packages": absent, "pinned": pinned, "observed": observed,
                           "version_mismatches": wrong}
        if absent or wrong:
            entry["status"] = "absent" if absent else "mismatch"


def check_saved_runs(report):
    with report.stage("saved_runs") as entry:
        detail = {}
        for label, relative in RUNS.items():
            coverage = manifest_coverage(ROOT / relative)
            missing = coverage.get("missing", [])
            detail[label] = {"declared": coverage["declared"], "missing": len(missing),
                             "missing_examples": missing[:6],
                             "missing_all_parquet": all(name.endswith(".parquet") for name in missing)}
        entry["detail"] = detail
        if any(item["missing"] for item in detail.values()):
            entry["status"] = "incomplete"


def time_input_hashing(report, config):
    """Measure `snapshot_inputs`, which D11 runs once before and once after a run."""
    with report.stage("input_hash_cost") as entry:
        from src.d11_io import snapshot_inputs

        snapshot = snapshot_inputs(ROOT, config)
        total = sum(item["bytes"] for item in snapshot.values())
        entry["detail"] = {"files": len(snapshot), "bytes": total,
                           "gigabytes": round(total / 1024 ** 3, 3),
                           "note": "a registered run pays this twice, before and after"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["local", "remote"], required=True,
                        help="local = the machine holding the data; remote = an agent container")
    parser.add_argument("--hash-inputs", action="store_true",
                        help="also time snapshot_inputs; needs the data and takes minutes")
    parser.add_argument("--out", default=None,
                        help="default: reports/d12/preflight_<scope>.json")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    baseline = json.loads((ROOT / "config/d11_baseline.json").read_text())

    report = Report()
    started = time.perf_counter()
    check_repository(report, baseline)
    check_inputs(report, config)
    check_registered_digests(report)
    check_environment(report, config)
    check_saved_runs(report)
    if args.hash_inputs:
        time_input_hashing(report, config)

    payload = {"scope": args.scope, "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "total_seconds": round(time.perf_counter() - started, 3),
               "ready_for_a_registered_run": not report.blocking, "stages": report.stages}
    out = Path(args.out) if args.out else ROOT / f"reports/d12/preflight_{args.scope}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    for entry in report.stages:
        print(f"  [{entry['status']:>10}] {entry['stage']:<18} {entry['seconds']:>8.3f}s")
    if report.blocking:
        print("\nblocking stages, in the order they must be cleared:")
        for entry in report.blocking:
            print(f"  {entry['stage']}: {json.dumps(entry['detail'], ensure_ascii=False)[:300]}")
        return 1
    print("\nall stages ok: this checkout may run a registered D12 experiment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

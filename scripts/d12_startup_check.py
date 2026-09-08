"""D12 startup dependency check (第1天 上午). Standard library only, read-only.

It must run before the scientific stack is installed and must never touch a D11 output.
Run it on the machine that holds the data and keep the TSV as the day-1 evidence; the
`--scope` column is what stops a remote container's empty checkout from being read as a
verdict about your machine.

    python3 scripts/d12_startup_check.py --scope local

The substantive checks live in `src/d12_repo.py`, shared with `scripts/d12_preflight.py`,
so the two entry points cannot drift apart about what "ready" means.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repo import (REQUIRED_PACKAGES, git, installed_versions, manifest_coverage,
                          missing_packages, pending_code_changes, protected_blob_drift,
                          registered_sha256, required_inputs, sha256)

FIELDS = ["category", "item", "required_by", "expected", "observed", "status", "blocks", "recovery"]
RUNS = {"d11_v1 (source run)": "reports/d11/d11_v1",
        "d11_diagnostic_v1": "reports/d11/d11_diagnostic_v1"}


def check_repository(add, baseline):
    branch, head = git(ROOT, "branch", "--show-current", check=False), git(ROOT, "rev-parse", "HEAD", check=False)
    add("repository", "branch", "src/d11_io.verify_repository", "d11 (D11 entry) / d12 (D12 work)",
        branch or "unknown", "ok" if branch else "unknown",
        "d11_diagnose.py refuses any branch but d11" if branch != "d11" else "",
        "D12 uses scripts/d12_preflight.py; do not delete the D11 branch guard")
    add("repository", "HEAD", "R0 identity", "663104bda3ac3bcb7cd2a8b529849c24f94efc5c",
        head, "ok" if head else "unknown", "", "")
    for name, revision, key in (("frozen-oos commit", "frozen-oos^{commit}", "frozen_commit"),
                                ("frozen-oos tag object", "frozen-oos", "frozen_tag_object")):
        actual = git(ROOT, "rev-parse", revision, check=False)
        add("repository", name, "src/d11_io.verify_repository", baseline[key],
            actual or "UNRESOLVED", "ok" if actual == baseline[key] else "missing",
            "" if actual else "preflight aborts", "" if actual else "git fetch --tags origin")
    drift = protected_blob_drift(ROOT, baseline)
    total = len(baseline["protected_blobs"])
    add("repository", "protected tracked files", "src/d11_io.verify_repository",
        f"{total} files unchanged", f"{total - len(drift)} intact, {len(drift)} altered",
        "ok" if not drift else "altered", ", ".join(drift[:5]),
        "" if not drift else "restore from base_commit")
    pending = pending_code_changes(ROOT)
    add("repository", "uncommitted code/config", "src/d11_io.verify_repository", "none",
        f"{len(pending)} path(s)", "ok" if not pending else "dirty", ", ".join(pending[:5]),
        "commit or stash before any registered run")


def check_inputs(add, config):
    required = required_inputs(config)
    absent = [path for path in required if not (ROOT / path).is_file()]
    add("source input", "config-declared required_inputs", "src/d11_io.preflight",
        f"{len(required)} files", f"{len(required) - len(absent)} present, {len(absent)} absent",
        "ok" if not absent else "absent", "preflight raises DiagnosticError" if absent else "",
        "restore data/ from the machine that ran D9-D11" if absent else "")
    provider = Path(config["data"]["qlib_provider_uri"]).expanduser()
    ready = (provider / "features").is_dir() and (provider / "calendars/day.txt").is_file()
    add("source input", "qlib provider_uri", "src/d11_io.preflight", str(provider),
        "present" if ready else "absent", "ok" if ready else "absent",
        "" if ready else "no features or labels can be rebuilt",
        "" if ready else "provide the cn_data_community bundle")


def check_runs(add):
    for label, relative in RUNS.items():
        coverage = manifest_coverage(ROOT / relative)
        missing = coverage["missing"]
        parquet = [name for name in missing if name.endswith(".parquet")]
        add("registered run", label, "R0 replay / P0 drill-down", f"{coverage['declared']} manifest outputs",
            f"{coverage['declared'] - len(missing)} present, {len(missing)} missing ({len(parquet)} parquet)",
            "ok" if not missing else "incomplete",
            "; ".join(missing[:4]) + (" ..." if len(missing) > 4 else ""),
            "copy from the machine that produced the run" if missing else "")


def check_digests(add):
    for relative, expected in registered_sha256(ROOT).items():
        path = ROOT / relative
        if not path.is_file():
            add("registered sha256", relative, "src/d11_io.preflight", expected[:16], "file absent",
                "absent", "preflight raises DiagnosticError", "restore the file, then re-verify")
            continue
        actual = sha256(path)
        add("registered sha256", relative, "src/d11_io.preflight", expected[:16], actual[:16],
            "ok" if actual == expected else "mismatch",
            "" if actual == expected else "preflight aborts", "")


def check_environment(add, summary):
    absent = missing_packages()
    add("environment", "python packages", "src/d11_io.preflight", f"{len(REQUIRED_PACKAGES)} importable",
        f"{len(REQUIRED_PACKAGES) - len(absent)} importable, {len(absent)} absent",
        "ok" if not absent else "absent", ", ".join(absent),
        "install the pinned versions recorded in reports/d9_v2_summary.json")
    pinned = summary["repository"]["dependency_versions"]
    for package, actual in installed_versions().items():
        expected = pinned[package]
        add("environment", f"{package} version pin", "src/d11_io.preflight", expected, actual,
            "ok" if actual == expected else "absent" if actual == "not installed" else "mismatch",
            "" if actual == expected else "preflight rejects a different version", "")


def identity_record(scope, baseline, rows):
    """What this machine can prove about R0's identity, kept separate from what it cannot."""
    diagnostic = ROOT / "reports/d11/d11_diagnostic_v1"
    source_manifest = ROOT / "reports/d11/d11_v1/manifest.json"
    declared = json.loads((diagnostic / "manifest.json").read_text()).get("source_manifest_sha256")
    return {
        "record_scope": scope, "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": {"branch": git(ROOT, "branch", "--show-current", check=False),
                       "head": git(ROOT, "rev-parse", "HEAD", check=False),
                       "base_commit": baseline["base_commit"],
                       "frozen_oos_commit": git(ROOT, "rev-parse", "frozen-oos^{commit}", check=False),
                       "protected_tracked_files": len(baseline["protected_blobs"]),
                       "protected_files_intact": not protected_blob_drift(ROOT, baseline),
                       "uncommitted_code_or_config": pending_code_changes(ROOT)},
        "r0": {"source_run": "d11_v1", "diagnostic_run": "d11_diagnostic_v1",
               "config_sha256": sha256(ROOT / "config/config.yaml"),
               "source_manifest_sha256_declared_by_diagnostic": declared,
               "source_manifest_sha256_recomputed":
                   sha256(source_manifest) if source_manifest.is_file() else None,
               "coverage": {label: {"declared": coverage["declared"],
                                    "missing": len(coverage["missing"]),
                                    "missing_all_parquet":
                                        all(name.endswith(".parquet") for name in coverage["missing"])}
                            for label, relative in RUNS.items()
                            for coverage in [manifest_coverage(ROOT / relative)]}},
        "blocking_rows": [dict(row) for row in rows if row["status"] != "ok"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["local", "remote"], required=True,
                        help="local = the machine holding the data; remote = an agent container")
    parser.add_argument("--out", default="reports/d12/startup_check.tsv")
    parser.add_argument("--identity-out", default=None,
                        help="also write the machine-verified R0 identity, e.g. "
                             "reports/d12/baseline_identity_local.json")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    baseline = json.loads((ROOT / "config/d11_baseline.json").read_text())
    summary = json.loads((ROOT / "reports/d9_v2_summary.json").read_text())

    rows: list[dict] = []
    add = lambda *values: rows.append(dict(zip(FIELDS, values)))
    check_repository(add, baseline)
    check_inputs(add, config)
    check_runs(add)
    check_digests(add)
    check_environment(add, summary)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scope", *FIELDS], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({"scope": args.scope, **row})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"wrote {out} ({len(rows)} rows, scope={args.scope})")
    print("status counts:", json.dumps(counts, ensure_ascii=False))
    for row in (row for row in rows if row["status"] != "ok"):
        print(f"  [{row['status']}] {row['category']} / {row['item']}: {row['observed']}")

    if args.identity_out:
        identity = ROOT / args.identity_out
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text(json.dumps(identity_record(args.scope, baseline, rows),
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

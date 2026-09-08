"""D12 startup dependency check (第1天 上午).

Standard library only, read-only: it must run before the scientific stack is
installed and must never touch D11 outputs. Run it on the machine that holds
the data, then keep reports/d12/startup_check.tsv as the day-1 evidence.

    python3 scripts/d12_startup_check.py --scope local
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml", "matplotlib",
            "qlib", "lightgbm", "cvxpy", "clarabel", "tushare"]


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(path):
    content = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def required_inputs(config):
    """Mirror of src.d11_io.required_inputs, kept import-free on purpose."""
    result = [Path(config["data"]["window_marker"])]
    for directory in [config["data"]["processed_design_dir"], config["d9"]["v2"]["processed_oos_dir"]]:
        result.extend(Path(directory) / f"{name}.parquet"
                      for name in ["benchmark_weights", "tradability_flags"])
    result.extend(Path(config["d6"][key]) for key in ["industry_mapping_file", "industry_catalog_file"])
    raw = Path(config["data"]["tushare_raw_dir"])
    result.extend(raw / name for name in ["index_daily_H00905_CSI.parquet", "trade_cal.parquet"])
    start_year = int(str(config["windows"]["oos_start"])[:4])
    end_year = int(str(config["windows"]["oos_end"])[:4])
    for year in range(start_year - 2, end_year + 1):
        result.append(raw / f"daily_basic_{year}.parquet")
    for year in range(start_year, end_year + 1):
        result.extend(raw / f"{endpoint}_{year}.parquet" for endpoint in ["stk_limit", "suspend_d"])
    result.append(Path("reports/d9_v2_feasibility_dates.parquet"))
    return result


def manifest_coverage(run_dir):
    """Present/missing split of a run manifest's declared outputs."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    outputs = manifest["output_sha256"]
    missing = sorted(name for name in outputs if not (run_dir / name).is_file())
    return len(outputs), missing


def check_repository(rows, baseline):
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    add = lambda *a: rows.append(dict(zip(FIELDS, a)))
    add("repository", "branch", "src/d11_io.verify_repository", "d11 (D11 entry) / d12 (D12 work)",
        branch or "unknown", "ok" if branch else "unknown",
        "d11_diagnose.py refuses any branch but d11" if branch != "d11" else "",
        "D12 needs its own entry point; do not delete the D11 branch guard")
    add("repository", "HEAD", "R0 identity", "663104bda3ac3bcb7cd2a8b529849c24f94efc5c",
        head, "ok" if head else "unknown", "", "")
    for name, rev in (("frozen-oos commit", "frozen-oos^{commit}"), ("frozen-oos tag object", "frozen-oos")):
        key = "frozen_commit" if "commit" in name else "frozen_tag_object"
        actual = git("rev-parse", rev)
        add("repository", name, "src/d11_io.verify_repository", baseline[key],
            actual or "UNRESOLVED", "ok" if actual == baseline[key] else "missing",
            "" if actual else "preflight aborts", "" if actual else "git fetch --tags origin")
    protected = baseline["protected_blobs"]
    broken = [rel for rel, exp in protected.items()
              if not (ROOT / rel).is_file() or (ROOT / rel).is_symlink() or git_blob(ROOT / rel) != exp]
    add("repository", "protected tracked files", "src/d11_io.verify_repository",
        f"{len(protected)} files unchanged", f"{len(protected) - len(broken)} intact, {len(broken)} altered",
        "ok" if not broken else "altered", ", ".join(broken[:5]), "" if not broken else "restore from base_commit")
    dirty = git("diff", "--name-only", "HEAD", "--", "src", "scripts", "tests", "config")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "tests", "config")
    pending = [p for p in (dirty + "\n" + untracked).split("\n") if p]
    add("repository", "uncommitted code/config", "src/d11_io.verify_repository", "none",
        f"{len(pending)} path(s)", "ok" if not pending else "dirty", ", ".join(pending[:5]),
        "commit or stash before any registered run")


FIELDS = ["category", "item", "required_by", "expected", "observed", "status", "blocks", "recovery"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["local", "remote"], required=True,
                        help="local = the machine holding the data; remote = an agent container")
    parser.add_argument("--out", default="reports/d12/startup_check.tsv")
    args = parser.parse_args()

    import yaml  # available in every environment that ever ran D1-D11
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    baseline = json.loads((ROOT / "config/d11_baseline.json").read_text())

    rows: list[dict] = []
    check_repository(rows, baseline)
    add = lambda *a: rows.append(dict(zip(FIELDS, a)))

    # --- source inputs -----------------------------------------------------
    required = required_inputs(config)
    absent = [p for p in required if not (ROOT / p).is_file()]
    add("source input", "config-declared required_inputs", "src/d11_io.preflight",
        f"{len(required)} files", f"{len(required) - len(absent)} present, {len(absent)} absent",
        "ok" if not absent else "absent", "preflight raises DiagnosticError" if absent else "",
        "restore data/ from the machine that ran D9-D11" if absent else "")
    provider = Path(config["data"]["qlib_provider_uri"]).expanduser()
    have_provider = (provider / "features").is_dir() and (provider / "calendars/day.txt").is_file()
    add("source input", "qlib provider_uri", "src/d11_io.preflight", str(provider),
        "present" if have_provider else "absent", "ok" if have_provider else "absent",
        "" if have_provider else "no features/labels can be rebuilt",
        "" if have_provider else "provide the cn_data_community bundle")

    # --- registered runs ---------------------------------------------------
    for label, run_dir in (("d11_v1 (source run)", ROOT / "reports/d11/d11_v1"),
                           ("d11_diagnostic_v1", ROOT / "reports/d11/d11_diagnostic_v1")):
        total, missing = manifest_coverage(run_dir)
        parquet = [m for m in missing if m.endswith(".parquet")]
        add("registered run", label, "R0 replay / P0 drill-down", f"{total} manifest outputs",
            f"{total - len(missing)} present, {len(missing)} missing ({len(parquet)} parquet)",
            "ok" if not missing else "incomplete",
            "; ".join(missing[:4]) + (" ..." if len(missing) > 4 else ""),
            "copy from the machine that produced the run" if missing else "")

    # --- registered SHA256 -------------------------------------------------
    manifest = json.loads((ROOT / "reports/d9_v2_manifest.json").read_text())
    summary = json.loads((ROOT / "reports/d9_v2_summary.json").read_text())
    checks = {"config/config.yaml": manifest["v2_config_sha256"],
              manifest["original_manifest"]: manifest["original_manifest_sha256"],
              manifest["recovery_manifest"]: manifest["recovery_manifest_sha256"],
              manifest["feasibility_audit"]: manifest["feasibility_audit_sha256"],
              "reports/d9_v2_feasibility_dates.parquet":
                  summary["v2_source_failures"]["feasibility_dates_sha256"]}
    for path, expected in checks.items():
        if not (ROOT / path).is_file():
            add("registered sha256", path, "src/d11_io.preflight", expected[:16], "file absent",
                "absent", "preflight raises DiagnosticError", "restore the file, then re-verify")
        else:
            actual = sha256(ROOT / path)
            add("registered sha256", path, "src/d11_io.preflight", expected[:16], actual[:16],
                "ok" if actual == expected else "mismatch",
                "" if actual == expected else "preflight aborts", "")

    # --- environment -------------------------------------------------------
    absent_pkgs = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    add("environment", "python packages", "src/d11_io.preflight", f"{len(PACKAGES)} importable",
        f"{len(PACKAGES) - len(absent_pkgs)} importable, {len(absent_pkgs)} absent",
        "ok" if not absent_pkgs else "absent", ", ".join(absent_pkgs),
        "install the pinned versions recorded in reports/d9_v2_summary.json")
    for package, expected in summary["repository"]["dependency_versions"].items():
        module = {"qlib": "qlib"}.get(package, package)
        try:
            import importlib.metadata as md
            actual = md.version("pyqlib" if package == "qlib" else package)
        except Exception:
            actual = "not installed"
        add("environment", f"{package} version pin", "src/d11_io.preflight", expected, actual,
            "ok" if actual == expected else "absent" if actual == "not installed" else "mismatch",
            "" if actual == expected else "preflight rejects a different version", "")

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
    blocking = [r for r in rows if r["status"] not in {"ok"}]
    for row in blocking:
        print(f"  [{row['status']}] {row['category']} / {row['item']}: {row['observed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D12 repository identity and input inventory. Standard library only, read-only.

Two rules shape this module.

First, D11's accepted guard is left exactly as it is. `src.d11_io.verify_repository`
still demands the `d11` branch and `scripts/d11_diagnose.py` still refuses to run
anywhere else; the week's plan forbids relaxing that check so a D12 run can pass. D12
therefore carries its own guard here, with the same substantive checks and a D12 branch
allow-list, rather than a weakened shared one.

Second, it must import before numpy/pandas/qlib exist, because the first thing a D12
day-1 check has to answer is whether they are installed at all. Nothing here imports
outside the standard library, and nothing here writes into `reports/d11`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

D12_BRANCHES = ("d12",)
#: Same list `src.d11_io.preflight` walks, so a "missing package" verdict means the
#: same thing on both entry points.
REQUIRED_PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml",
                     "matplotlib", "qlib", "lightgbm", "cvxpy", "clarabel", "tushare")
#: Distribution names for the four versions `src.d11_io.preflight` pins exactly.
PINNED_DISTRIBUTIONS = {"qlib": "pyqlib", "lightgbm": "lightgbm",
                        "cvxpy": "cvxpy", "clarabel": "clarabel"}


class D12Error(RuntimeError):
    """A D12 check failed. Deliberately not `DiagnosticError`, which needs numpy."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(path):
    content = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def git(root, *args, check=True):
    """`check=False` returns '' instead of raising, for checks that must keep going."""
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode and check:
        raise D12Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip() if not result.returncode else ""


def required_inputs(config):
    """Mirror of `src.d11_io.required_inputs`, kept import-free on purpose.

    `tests/test_d12_core.py::test_required_inputs_matches_d11` pins the two together, so
    the mirror cannot drift away from the list D11's preflight actually enforces.
    """
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


def registered_sha256(root):
    """The five files `src.d11_io.preflight` checks against historically fixed digests."""
    reports = Path(root) / "reports"
    manifest = json.loads((reports / "d9_v2_manifest.json").read_text())
    summary = json.loads((reports / "d9_v2_summary.json").read_text())
    return {"config/config.yaml": manifest["v2_config_sha256"],
            manifest["original_manifest"]: manifest["original_manifest_sha256"],
            manifest["recovery_manifest"]: manifest["recovery_manifest_sha256"],
            manifest["feasibility_audit"]: manifest["feasibility_audit_sha256"],
            "reports/d9_v2_feasibility_dates.parquet":
                summary["v2_source_failures"]["feasibility_dates_sha256"]}


def manifest_coverage(run_dir):
    """Declared/present/missing split of a saved run, without reading the payloads."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"declared": 0, "missing": [], "run_dir_present": run_dir.is_dir(),
                "manifest_present": False}
    outputs = json.loads(manifest_path.read_text())["output_sha256"]
    return {"declared": len(outputs), "manifest_present": True, "run_dir_present": True,
            "missing": sorted(name for name in outputs if not (run_dir / name).is_file())}


def protected_blob_drift(root, baseline):
    """Tracked D1-D10 files whose bytes no longer match the accepted D11 baseline."""
    root = Path(root)
    return [relative for relative, expected in baseline["protected_blobs"].items()
            if not (root / relative).is_file() or (root / relative).is_symlink()
            or git_blob(root / relative) != expected]


def pending_code_changes(root):
    """Uncommitted or untracked code/config, which D11 rejects and D12 rejects too."""
    tracked = git(root, "diff", "--name-only", "HEAD", "--", "src", "scripts", "tests", "config", check=False)
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "tests", "config", check=False)
    return [path for path in (tracked + "\n" + untracked).split("\n") if path]


def verify_repository(root, baseline, *, branches=D12_BRANCHES):
    """`src.d11_io.verify_repository`'s checks, with a D12 branch allow-list.

    Every other condition is kept identical and in the same order, so a D12 run that
    passes here would also have passed D11's guard had it been on the `d11` branch.
    """
    root = Path(root)
    branch = git(root, "branch", "--show-current")
    if branch not in branches:
        raise D12Error(f"D12 requires one of {branches}; actual={branch!r}")
    frozen = git(root, "rev-parse", "frozen-oos^{commit}")
    tag_object = git(root, "rev-parse", "frozen-oos")
    if frozen != baseline["frozen_commit"] or tag_object != baseline["frozen_tag_object"]:
        raise D12Error("frozen-oos tag/commit changed; run `git fetch --tags origin` on a fresh clone")
    git(root, "merge-base", "--is-ancestor", baseline["base_commit"], "HEAD")
    # core.quotePath would escape the Chinese filenames, as it does in src/d11_io.py.
    tree = git(root, "-c", "core.quotePath=false", "ls-tree", "-r", baseline["base_commit"])
    accepted = {line.split("\t", 1)[1]: line.split()[2] for line in tree.splitlines()
                if line.split()[1] == "blob"}
    if accepted != baseline["protected_blobs"]:
        raise D12Error("D11 baseline file differs from actual accepted Git tree")
    drift = protected_blob_drift(root, baseline)
    if drift:
        raise D12Error("D1-D10 tracked files differ from accepted d11 baseline: " + ", ".join(drift))
    pending = pending_code_changes(root)
    if pending:
        raise D12Error("uncommitted code/config changes; commit on d12 before a registered run: "
                       + ", ".join(pending[:10]))
    return {"branch": branch, "head": git(root, "rev-parse", "HEAD"),
            "baseline": baseline["base_commit"], "frozen_oos": frozen,
            "frozen_tag_object": tag_object,
            "protected_tracked_files": len(baseline["protected_blobs"])}


def installed_versions():
    """Observed versions of the four pinned distributions; absent ones say so."""
    import importlib.metadata

    observed = {}
    for package, distribution in PINNED_DISTRIBUTIONS.items():
        try:
            observed[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = "not installed"
    return observed


def missing_packages():
    import importlib.util

    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]

"""Read-only inputs and exclusive, isolated D10 output namespace."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
from pathlib import Path

from src.d10_core import DiagnosticError


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(path):
    content = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def verify_repository(root, baseline):
    branch = git(root, "branch", "--show-current")
    if branch != "d10":
        raise DiagnosticError(f"D10 requires d10 branch; actual={branch!r}")
    frozen = git(root, "rev-parse", "frozen-oos^{commit}")
    tag_object = git(root, "rev-parse", "frozen-oos")
    if frozen != baseline["frozen_commit"] or tag_object != baseline["frozen_tag_object"]:
        raise DiagnosticError("frozen-oos tag/commit changed")
    git(root, "merge-base", "--is-ancestor", baseline["base_commit"], "HEAD")
    tree = git(root, "ls-tree", "-r", baseline["base_commit"])
    expected_tree = {line.split("\t", 1)[1]: line.split()[2] for line in tree.splitlines()
                     if line.split()[1] == "blob"}
    # core.quotePath would escape Chinese names. Request an unquoted tree instead.
    if expected_tree != baseline["protected_blobs"]:
        raw_tree = git(root, "-c", "core.quotePath=false", "ls-tree", "-r", baseline["base_commit"])
        expected_tree = {line.split("\t", 1)[1]: line.split()[2] for line in raw_tree.splitlines()
                         if line.split()[1] == "blob"}
    if expected_tree != baseline["protected_blobs"]:
        raise DiagnosticError("D10 baseline file differs from actual accepted Git tree")
    changed = []
    for relative, expected in baseline["protected_blobs"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or git_blob(path) != expected:
            changed.append(relative)
    if changed:
        raise DiagnosticError("D1-D9 tracked files differ from accepted d10 baseline: " + ", ".join(changed))
    # D10 code must also be committed, so run manifests identify reproducible code.
    if git(root, "diff", "--name-only", "HEAD", "--", "src", "scripts", "tests", "config"):
        raise DiagnosticError("uncommitted code/config changes; commit only on d10 before running")
    return {"branch": branch, "head": git(root, "rev-parse", "HEAD"),
            "baseline": baseline["base_commit"], "frozen_oos": frozen,
            "frozen_tag_object": tag_object, "protected_tracked_files": len(baseline["protected_blobs"])}


def required_inputs(config):
    reports = Path(config["data"]["reports_dir"])
    result = [reports / name for name in ["d8_summary.json", "d9_v2_summary.json", "d9_v2_manifest.json",
        "d9_run_manifest.json", "d9_recovery_manifest.json", "d9_v2_feasibility_audit.json",
        "d9_v2_feasibility_dates.parquet", "d9_v2_main_daily.parquet", "d9_v2_main_monthly.parquet",
        "d9_v2_impact2x_daily.parquet", "d9_v2_impact2x_trades.parquet", "d6_factor_exposures.parquet",
        "d9_v2_factor_exposures.parquet", "d7_target_weights.parquet", "d9_v2_target_weights.parquet",
        "d8_target_audit.parquet", "d9_v2_ladder_target_audit.parquet",
        "d8_ladder_summary.parquet", "d9_v2_ladder_summary.parquet",
        "d5_walk_forward_predictions.parquet", "d9_v2_oos_predictions.parquet"]]
    for prefix in ["d8", "d9_v2"]:
        for layer in range(7):
            result.extend(reports / f"{prefix}_layer_{layer}_{suffix}.parquet" for suffix in ["targets", "daily", "trades"])
    for directory in [config["data"]["processed_design_dir"], config["d9"]["v2"]["processed_oos_dir"]]:
        result.extend(Path(directory) / f"{name}.parquet" for name in ["benchmark_weights", "tradability_flags"])
    result.extend(Path(config["d6"][k]) for k in ["industry_mapping_file", "industry_catalog_file"])
    raw = Path(config["data"]["tushare_raw_dir"])
    result.extend(raw / name for name in ["index_daily_H00905_CSI.parquet", "trade_cal.parquet"])
    end_year = int(str(config["windows"]["oos_end"])[:4])
    for year in range(2014, end_year + 1):
        result.extend(raw / f"{endpoint}_{year}.parquet" for endpoint in ["daily_basic", "stk_limit", "suspend_d"])
    result.append(Path(config["data"]["window_marker"]))
    return result


def preflight(root, config, baseline):
    repository = verify_repository(root, baseline)
    missing = [str(p) for p in required_inputs(config) if not (root / p).is_file()]
    provider = Path(config["data"]["qlib_provider_uri"]).expanduser()
    if not (provider / "features").is_dir() or not (provider / "calendars/day.txt").is_file():
        missing.append(str(provider))
    dependencies = [m for m in ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml", "matplotlib", "qlib"]
                    if importlib.util.find_spec(m) is None]
    if missing or dependencies:
        raise DiagnosticError(f"Missing local inputs: {missing}; missing packages: {dependencies}. Do not rerun D1-D9 to repair automatically.")
    marker = json.loads((root / config["data"]["window_marker"]).read_text())
    if marker.get("mode") != "oos":
        raise DiagnosticError("D10 expects existing D9 oos marker; it will not unseal or modify the marker")
    reports = root / config["data"]["reports_dir"]
    manifest = json.loads((reports / "d9_v2_manifest.json").read_text())
    summary = json.loads((reports / "d9_v2_summary.json").read_text())
    if (manifest.get("status") != "completed" or manifest.get("decision") != "No-Go"
            or summary["threshold_evaluation"]["decision"] != "No-Go"):
        raise DiagnosticError("Expected completed D9-V2 No-Go evidence")
    checks = {"config/config.yaml": manifest["v2_config_sha256"],
              manifest["original_manifest"]: manifest["original_manifest_sha256"],
              manifest["recovery_manifest"]: manifest["recovery_manifest_sha256"],
              manifest["feasibility_audit"]: manifest["feasibility_audit_sha256"],
              "reports/d9_v2_feasibility_dates.parquet": summary["v2_source_failures"]["feasibility_dates_sha256"]}
    for path, expected in checks.items():
        if sha256(root / path) != expected:
            raise DiagnosticError(f"Historical registered SHA256 mismatch: {path}")
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "pyqlib"]:
        versions[package] = importlib.metadata.version(package)
    expected_qlib = summary["repository"]["dependency_versions"]["qlib"]
    if versions["pyqlib"] != expected_qlib:
        raise DiagnosticError(f"Use original Qlib {expected_qlib}; observed {versions['pyqlib']}")
    return {"repository": repository, "versions": versions, "decision": "No-Go", "preflight_passed": True}


def snapshot_inputs(root, config):
    """SHA256 protected inputs, including raw provider bins, before AND after.

    Qlib cache directories are excluded because expression evaluation may
    generate caches. Original feature bins/calendar/instruments are included.
    """
    root = Path(root).resolve()
    paths = set()
    for directory in ["config", "src", "scripts", "tests", "reports", "models", "data"]:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or "__pycache__" in path.parts or relative.parts[:2] == ("reports", "d10"):
                continue
            paths.add(path)
    provider = Path(config["data"]["qlib_provider_uri"]).expanduser().resolve()
    for directory in ["features", "calendars", "instruments"]:
        paths.update(p for p in (provider / directory).rglob("*") if p.is_file())
    output = {}
    for index, path in enumerate(sorted(paths), 1):
        if index % 5000 == 0:
            print(f"D10 input SHA256 {index}/{len(paths)}", flush=True)
        key = str(path.relative_to(root)) if path.is_relative_to(root) else "qlib/" + str(path.relative_to(provider))
        output[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return output


def assert_snapshot_unchanged(before, after):
    changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
    if changed:
        raise DiagnosticError("Protected inputs changed during D10: " + ", ".join(changed[:30]))


class D10Output:
    def __init__(self, root, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise DiagnosticError("run-id must be a simple version name, no path separators")
        root = Path(root).resolve()
        parent = root / "reports" / "d10"
        if parent.resolve() != parent or (parent / run_id).is_symlink():
            raise DiagnosticError("D10 output path may not traverse a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / run_id
        self.path.mkdir(exist_ok=False)

    def target(self, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
            raise DiagnosticError("output filename must stay inside D10 run directory")
        target = self.path / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        return target

    def json(self, name, payload):
        with self.target(name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            stream.write("\n")

    def text(self, name, text):
        with self.target(name).open("x", encoding="utf-8") as stream:
            stream.write(text)

    def parquet(self, name, frame):
        with self.target(name).open("xb") as stream:
            frame.to_parquet(stream, index=False)

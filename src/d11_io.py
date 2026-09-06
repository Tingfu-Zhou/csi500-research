"""Read-only inputs and exclusive, isolated D11 output namespace."""
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
    if branch != "d11":
        raise DiagnosticError(f"D11 requires d11 branch; actual={branch!r}")
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
        raise DiagnosticError("D11 baseline file differs from actual accepted Git tree")
    changed = []
    for relative, expected in baseline["protected_blobs"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or git_blob(path) != expected:
            changed.append(relative)
    if changed:
        raise DiagnosticError("D1-D10 tracked files differ from accepted d11 baseline: " + ", ".join(changed))
    # D11 code must also be committed, so run manifests identify reproducible code.
    if (git(root, "diff", "--name-only", "HEAD", "--", "src", "scripts", "tests", "config")
            or git(root, "ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "tests", "config")):
        raise DiagnosticError("uncommitted code/config changes; commit only on d11 before running")
    return {"branch": branch, "head": git(root, "rev-parse", "HEAD"),
            "baseline": baseline["base_commit"], "frozen_oos": frozen,
            "frozen_tag_object": tag_object, "protected_tracked_files": len(baseline["protected_blobs"])}


def required_inputs(config):
    result = [Path(config['data']['window_marker'])]
    for directory in [config['data']['processed_design_dir'], config['d9']['v2']['processed_oos_dir']]:
        result.extend(Path(directory) / f'{name}.parquet' for name in ['benchmark_weights', 'tradability_flags'])
    result.extend(Path(config['d6'][k]) for k in ['industry_mapping_file', 'industry_catalog_file'])
    raw = Path(config['data']['tushare_raw_dir'])
    result.extend(raw / name for name in ['index_daily_H00905_CSI.parquet', 'trade_cal.parquet'])
    start_year = int(str(config['windows']['oos_start'])[:4])
    end_year = int(str(config['windows']['oos_end'])[:4])
    for year in range(start_year - 2, end_year + 1):
        result.append(raw / f'daily_basic_{year}.parquet')
    for year in range(start_year, end_year + 1):
        result.extend(raw / f'{endpoint}_{year}.parquet' for endpoint in ['stk_limit', 'suspend_d'])
    result.append(Path('reports/d9_v2_feasibility_dates.parquet'))
    return result


def preflight(root, config, baseline):
    repository = verify_repository(root, baseline)
    missing = [str(p) for p in required_inputs(config) if not (root / p).is_file()]
    provider = (root / Path(config["data"]["qlib_provider_uri"]).expanduser())
    if not (provider / "features").is_dir() or not (provider / "calendars/day.txt").is_file():
        missing.append(str(provider))
    dependencies = [m for m in ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml", "matplotlib", "qlib", "lightgbm", "cvxpy", "clarabel", "tushare"]
                    if importlib.util.find_spec(m) is None]
    if missing or dependencies:
        raise DiagnosticError(f"Missing local inputs: {missing}; missing packages: {dependencies}. Do not rerun D1-D10 to repair automatically.")
    marker = json.loads((root / config["data"]["window_marker"]).read_text())
    if marker.get("mode") != "oos":
        raise DiagnosticError("D11 expects existing D9 oos marker; it will not unseal or modify the marker")
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
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "pyqlib", "lightgbm", "cvxpy", "clarabel"]:
        versions[package] = importlib.metadata.version(package)
    expected_qlib = summary["repository"]["dependency_versions"]["qlib"]
    if versions["pyqlib"] != expected_qlib:
        raise DiagnosticError(f"Use original Qlib {expected_qlib}; observed {versions['pyqlib']}")
    for package in ['lightgbm', 'cvxpy', 'clarabel']:
        expected = summary['repository']['dependency_versions'][package]
        if versions[package] != expected:
            raise DiagnosticError(f'Use original {package} {expected}; observed {versions[package]}')
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
            if not path.is_file() or "__pycache__" in path.parts or relative.parts[:2] == ("reports", "d11"):
                continue
            paths.add(path)
    provider = (root / Path(config["data"]["qlib_provider_uri"]).expanduser()).resolve()
    for directory in ["features", "calendars", "instruments"]:
        paths.update(p for p in (provider / directory).rglob("*") if p.is_file())
    output = {}
    for index, path in enumerate(sorted(paths), 1):
        if index % 5000 == 0:
            print(f"D11 input SHA256 {index}/{len(paths)}", flush=True)
        key = str(path.relative_to(root)) if path.is_relative_to(root) else "qlib/" + str(path.relative_to(provider))
        output[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return output


def assert_snapshot_unchanged(before, after):
    changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
    if changed:
        raise DiagnosticError("Protected inputs changed during D11: " + ", ".join(changed[:30]))


class D11Output:
    def __init__(self, root, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise DiagnosticError("run-id must be a simple version name, no path separators")
        root = Path(root).resolve()
        parent = root / "reports" / "d11"
        if parent.resolve() != parent or (parent / run_id).is_symlink():
            raise DiagnosticError("D11 output path may not traverse a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / run_id
        self.path.mkdir(exist_ok=False)

    def target(self, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
            raise DiagnosticError("output filename must stay inside D11 run directory")
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

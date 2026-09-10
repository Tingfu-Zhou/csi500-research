"""Immutable Day-6 outputs, Day-5 verification and complete input snapshots."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.d12_repo import D12Error, sha256


RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def clean(value):
    from scripts.d11_diagnose import clean as original
    return original(value)


class Day6Output:
    """One non-overwritable directory under ``reports/d12/day6``."""

    def __init__(self, root, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise D12Error("Use a simple new Day-6 run ID")
        root = Path(root).resolve()
        parent = root / "reports/d12/day6"
        if parent.resolve() != parent or (parent / run_id).is_symlink():
            raise D12Error("Day-6 output path may not traverse a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        self.root, self.run_id = root, run_id
        self.path = parent / run_id
        try:
            self.path.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise D12Error(
                f"Day-6 run {run_id!r} already exists and is immutable; "
                "do not delete or reuse it. Use a new --run-id only after a code fix."
            ) from error

    def target(self, name):
        if not FILE_NAME.fullmatch(name) or name in {".", ".."}:
            raise D12Error("Output filename must stay inside the Day-6 run")
        target = self.path / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        return target

    def json(self, name, value):
        self.target(name).write_text(
            json.dumps(clean(value), ensure_ascii=False, indent=2,
                       allow_nan=False, default=str) + "\n", encoding="utf-8")

    def text(self, name, value):
        self.target(name).write_text(value, encoding="utf-8")

    def parquet(self, name, frame):
        frame.to_parquet(self.target(name), index=False)

    def table(self, name, frame):
        self.text(name + ".tsv", frame.to_csv(sep="\t", index=False))

    def hashes(self):
        return {path.name: sha256(path) for path in sorted(self.path.iterdir())
                if path.is_file() and path.name != "manifest.json"}


def require_day5_run(root, run_id):
    if not RUN_ID.fullmatch(run_id):
        raise D12Error("Use a simple Day-5 run ID")
    path = Path(root).resolve() / "reports/d12/day5" / run_id
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise D12Error(f"Day-5 run {run_id!r} has no local manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed"
            or manifest.get("inputs_unchanged") is not True
            or manifest.get("mode") != "day5_diagnostics_and_stress"):
        raise D12Error(f"Day-5 run {run_id!r} is not a completed unchanged-input run")
    outputs = manifest.get("output_sha256", {})
    required = {"summary.json", "performance_matrix.tsv", "yearly_ic.tsv",
                "signal_calibration_deciles.tsv", "reference_gates.tsv",
                "repaired_r0_nominal_monthly_performance.parquet",
                "r1_nominal_monthly_performance.parquet"}
    if not required.issubset(outputs):
        raise D12Error("Day-5 manifest does not register every Day-6 prerequisite")
    for number, (name, digest) in enumerate(outputs.items(), 1):
        candidate = path / name
        if Path(name).name != name or not candidate.is_file() or sha256(candidate) != digest:
            raise D12Error(f"Day-5 output missing or changed: {name}")
        if number % 25 == 0:
            print(f"D12 Day6 Day5 SHA256 {number}/{len(outputs)}", flush=True)
    return path, manifest


def snapshot_inputs(root, config):
    """Hash Day-6 inputs, including Day 4/5 evidence, excluding own outputs."""
    root = Path(root).resolve()
    registered = (root / Path(config["data"]["qlib_provider_uri"]).expanduser()).resolve()
    paths = set()
    for directory in ["config", "src", "scripts", "tests", "reports", "models", "data"]:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            if relative.parts[:2] == ("reports", "d11"):
                continue
            if relative.parts[:3] == ("reports", "d12", "day6"):
                continue
            paths.add(path)
    for directory in ["features", "calendars", "instruments"]:
        base = registered / directory
        if base.exists():
            paths.update(path for path in base.rglob("*") if path.is_file())
    result = {}
    for number, path in enumerate(sorted(paths), 1):
        if number % 5000 == 0:
            print(f"D12 Day6 input SHA256 {number}/{len(paths)}", flush=True)
        key = (str(path.relative_to(root)) if path.is_relative_to(root)
               else "qlib/" + str(path.relative_to(registered)))
        result[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def assert_snapshot_unchanged(before, after):
    if before != after:
        names = sorted(set(before) | set(after))
        changed = [name for name in names if before.get(name) != after.get(name)]
        raise D12Error("Day-6 inputs changed during the run: " + ", ".join(changed[:20]))

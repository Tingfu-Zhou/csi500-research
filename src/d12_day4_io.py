"""Immutable Day-4 outputs and input snapshots for the registered R1 experiment."""
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


class Day4Output:
    """One non-overwritable directory under ``reports/d12/day4``."""

    def __init__(self, root, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise D12Error("Use a simple new Day-4 run ID")
        root = Path(root).resolve()
        parent = root / "reports/d12/day4"
        if parent.resolve() != parent or (parent / run_id).is_symlink():
            raise D12Error("Day-4 output path may not traverse a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.run_id = run_id
        self.path = parent / run_id
        self.path.mkdir(exist_ok=False)

    def target(self, name):
        if not FILE_NAME.fullmatch(name) or name in {".", ".."}:
            raise D12Error("Output filename must stay inside the Day-4 run")
        target = self.path / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        return target

    def json(self, name, value):
        self.target(name).write_text(
            json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
            encoding="utf-8")

    def text(self, name, value):
        self.target(name).write_text(value, encoding="utf-8")

    def parquet(self, name, frame):
        frame.to_parquet(self.target(name), index=False)

    def table(self, name, frame):
        self.text(name + ".tsv", frame.to_csv(sep="\t", index=False))

    def hashes(self):
        return {p.name: sha256(p) for p in sorted(self.path.iterdir())
                if p.is_file() and p.name != "manifest.json"}


def snapshot_inputs(root, config):
    """Hash every research input while excluding immutable run outputs.

    The corrected provider and rebuilt caches are inputs to Day 4 and are therefore
    included.  Only D11's immutable run namespace and Day 4's own output namespace
    are excluded.  This mirrors the earlier D11/D12 guards without allowing a run to
    hide changes to the repaired bundle under ``data/d12_repair``.
    """
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
            if relative.parts[:3] == ("reports", "d12", "day4"):
                continue
            paths.add(path)
    for directory in ["features", "calendars", "instruments"]:
        base = registered / directory
        if base.exists():
            paths.update(path for path in base.rglob("*") if path.is_file())
    result = {}
    for number, path in enumerate(sorted(paths), 1):
        if number % 5000 == 0:
            print(f"D12 Day4 input SHA256 {number}/{len(paths)}", flush=True)
        key = (str(path.relative_to(root)) if path.is_relative_to(root)
               else "qlib/" + str(path.relative_to(registered)))
        result[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def assert_snapshot_unchanged(before, after):
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [name for name in changed if before.get(name) != after.get(name)]
        raise D12Error("Day-4 inputs changed during the run: " + ", ".join(changed[:20]))


def require_day4_run(root, run_id):
    path = Path(root).resolve() / "reports/d12/day4" / run_id
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise D12Error(f"Day-4 run {run_id!r} has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("inputs_unchanged") is not True:
        raise D12Error(f"Day-4 run {run_id!r} did not complete with unchanged inputs")
    for name, digest in manifest.get("output_sha256", {}).items():
        candidate = path / name
        if Path(name).name != name or not candidate.is_file() or sha256(candidate) != digest:
            raise D12Error(f"Day-4 run output changed: {name}")
    return path, manifest

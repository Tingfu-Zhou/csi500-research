"""Immutable, stage-addressed outputs for the D12 price repair and repaired R0.

The repair runs in stages that each take minutes to hours, so unlike the day-3
entry point it cannot be a single all-or-nothing directory.  Each stage gets its
own subdirectory and its own manifest, and a later stage refuses to start until it
has re-verified the digests the earlier stage recorded.  A stage never rewrites a
directory that already exists: re-running one means choosing a new repair ID, the
same rule the D11 and day-3 runs already follow.

Nothing here writes into ``reports/d11``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.d12_repo import D12Error, sha256

RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}')
FILE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def clean(value):
    """The day-3 JSON normaliser, so numpy scalars and timestamps serialise alike."""
    from scripts.d11_diagnose import clean as original
    return original(value)


class StageOutput:
    """One stage directory under ``reports/d12/repair/<repair-id>/<stage>``."""

    def __init__(self, root, repair_id, stage, *, parent='reports/d12/repair'):
        if not RUN_ID.fullmatch(repair_id) or not RUN_ID.fullmatch(stage):
            raise D12Error('Repair and stage IDs must be simple names with no path separators')
        root = Path(root).resolve()
        base = root / parent / repair_id
        if base.resolve() != base or (base / stage).is_symlink():
            raise D12Error('Repair output path may not traverse a symlink')
        base.mkdir(parents=True, exist_ok=True)
        self.root, self.repair_id, self.stage = root, repair_id, stage
        self.path = base / stage
        self.path.mkdir(exist_ok=False)

    def target(self, name):
        if not FILE_NAME.fullmatch(name) or name in {'.', '..'}:
            raise D12Error('Output filename must stay inside the stage directory')
        target = self.path / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        return target

    def json(self, name, value):
        self.target(name).write_text(
            json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n',
            encoding='utf-8')

    def text(self, name, value):
        self.target(name).write_text(value, encoding='utf-8')

    def parquet(self, name, frame):
        frame.to_parquet(self.target(name), index=False)

    def table(self, name, frame):
        self.text(name + '.tsv', frame.to_csv(sep='\t', index=False))

    def hashes(self):
        return {p.name: sha256(p) for p in sorted(self.path.iterdir())
                if p.is_file() and p.name != 'manifest.json'}

    def finish(self, manifest):
        payload = dict(manifest)
        payload.update(repair_id=self.repair_id, stage=self.stage, output_sha256=self.hashes())
        self.json('manifest.json', payload)
        return payload


def stage_path(root, repair_id, stage, *, parent='reports/d12/repair'):
    return Path(root).resolve() / parent / repair_id / stage


def require_stage(root, repair_id, stage, *, parent='reports/d12/repair'):
    """Load a completed earlier stage and re-verify every file it recorded."""
    path = stage_path(root, repair_id, stage, parent=parent)
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file():
        raise D12Error(f'Stage {stage!r} has not completed for repair {repair_id!r}; run it first')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'completed':
        raise D12Error(f'Stage {stage!r} did not complete: {manifest.get("status")!r}')
    for name, digest in manifest.get('output_sha256', {}).items():
        candidate = path / name
        if Path(name).name != name or not candidate.is_file() or sha256(candidate) != digest:
            raise D12Error(f'Stage {stage!r} output changed after it completed: {name}')
    return path, manifest

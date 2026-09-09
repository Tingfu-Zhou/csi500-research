"""Build a corrected Qlib provider beside the original, never on top of it.

Alpha158 is evaluated inside Qlib's expression engine, straight off the provider
bins.  A correction applied to a loaded market frame therefore reaches the
simulator and misses the features, the labels and the risk exposures entirely.
The only place a factor repair can be applied once and be seen by every consumer
is the provider itself -- so this module writes a *second* provider and leaves the
registered one byte-identical.

Everything the corrected tree does not repair is hard-linked, so the copy costs
inodes rather than gigabytes.  Only ``features``, ``calendars`` and ``instruments``
are linked: exactly the three directories D11's input snapshot covers, and nothing
that Qlib might later write through a shared inode.

The ``.day.bin`` layout (one little-endian float32 calendar offset, then the
values) is not taken on faith.  ``prove_bin_decoding`` re-derives a series from the
bins and compares it against what Qlib itself returns for the same instrument,
field and dates, and the build refuses to start until that check passes.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.d12_repo import D12Error, sha256

LINKED_DIRECTORIES = ('features', 'calendars', 'instruments')
#: Adjusted quotes scale with the factor, volume scales inversely, and everything
#: else in the bundle stays exactly where it is.
SCALED_FIELDS = ('open', 'high', 'low', 'close', 'vwap', 'factor')
INVERSE_FIELDS = ('volume',)
BIN_NAME = re.compile(r'^(?P<field>[A-Za-z0-9_]+)\.day\.bin$')


def read_day_bin(path):
    """Return ``(calendar_start_index, float32 values)`` for one ``<field>.day.bin``."""
    raw = np.fromfile(Path(path), dtype='<f4')
    if raw.size < 1:
        raise D12Error(f'Empty Qlib bin: {path}')
    start = float(raw[0])
    if not np.isfinite(start) or start < 0 or start != np.floor(start):
        raise D12Error(f'{path}: leading value {start!r} is not a calendar offset')
    return int(start), raw[1:].copy()


def write_day_bin(path, start_index, values):
    payload = np.empty(len(values) + 1, dtype='<f4')
    payload[0] = np.float32(start_index)
    payload[1:] = np.asarray(values, dtype='<f4')
    Path(path).write_bytes(payload.tobytes())


def bin_series(path, calendar):
    """The (date, value) series a ``.day.bin`` encodes, on the provider calendar."""
    start, values = read_day_bin(path)
    calendar = pd.DatetimeIndex(calendar)
    if start + len(values) > len(calendar):
        raise D12Error(f'{path}: {len(values)} values from offset {start} overrun a {len(calendar)}-session calendar')
    return pd.Series(values, index=calendar[start:start + len(values)])


def read_provider_calendar(provider):
    path = Path(provider) / 'calendars/day.txt'
    if not path.is_file():
        raise D12Error(f'Missing provider calendar: {path}')
    dates = pd.to_datetime(pd.read_csv(path, header=None)[0]).dt.normalize()
    return pd.DatetimeIndex(dates).sort_values()


def instrument_directory(provider, instrument):
    """Qlib stores per-instrument directories lower-cased; project frames use upper."""
    return Path(provider) / 'features' / str(instrument).lower()


def prove_bin_decoding(provider, calendar, samples, reader, *, rtol=1e-6):
    """Check this module's bin decoding against Qlib's own reader before writing.

    ``samples`` are ``(instrument, field)`` pairs and ``reader(instrument, field,
    start, end)`` must return a date-indexed Series from Qlib.  A mismatch means the
    layout assumption is wrong for this bundle, and the caller must stop rather than
    patch files it cannot read correctly.
    """
    checks = []
    for instrument, field in samples:
        path = instrument_directory(provider, instrument) / f'{field}.day.bin'
        if not path.is_file():
            raise D12Error(f'Cannot prove bin decoding; missing {path}')
        decoded = bin_series(path, calendar).astype('float64')
        decoded = decoded[np.isfinite(decoded)]
        if decoded.empty:
            raise D12Error(f'{path} decoded to no finite values')
        served = reader(instrument, field, decoded.index.min(), decoded.index.max())
        served = pd.Series(served).astype('float64')
        served.index = pd.DatetimeIndex(served.index).normalize()
        shared = decoded.index.intersection(served.index)
        if len(shared) < max(2, int(.5 * len(served))):
            raise D12Error(f'{instrument}/{field}: bins and Qlib share only {len(shared)} sessions')
        left, right = decoded.reindex(shared), served.reindex(shared)
        both = np.isfinite(left) & np.isfinite(right)
        error = float(np.nanmax((left[both] / right[both] - 1).abs())) if both.any() else np.inf
        if not np.isfinite(error) or error > rtol:
            raise D12Error(f'{instrument}/{field}: bin decoding disagrees with Qlib by {error:.3e}')
        checks.append({'instrument': instrument, 'field': field, 'sessions_compared': int(both.sum()),
                       'maximum_relative_error': error})
    if not checks:
        raise D12Error('Bin decoding must be proven on at least one instrument')
    return checks


def _link_tree(source, target, *, mode):
    """Mirror the three provider directories; return per-file link bookkeeping."""
    linked = 0
    for directory in LINKED_DIRECTORIES:
        origin = Path(source) / directory
        if not origin.is_dir():
            raise D12Error(f'Source provider is missing {directory}/: {origin}')
        for path in sorted(origin.rglob('*')):
            if not path.is_file():
                continue
            destination = Path(target) / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise D12Error(f'Corrected provider path already exists: {destination}')
            if mode == 'link':
                try:
                    os.link(path, destination)
                except OSError:
                    shutil.copy2(path, destination)
            else:
                shutil.copy2(path, destination)
            linked += 1
    return linked


def build_corrected_provider(source, target, plan, *, mode='link', progress=None):
    """Mirror ``source`` into ``target`` and rewrite only the planned instrument bins.

    A patched file is unlinked before it is written, so the new content lands on a
    fresh inode and the original bundle's bytes cannot be touched through a shared
    one.
    """
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser()
    if mode not in {'link', 'copy'}:
        raise D12Error("mode must be 'link' or 'copy'")
    if target.exists():
        raise D12Error(f'Corrected provider directory already exists; use a new ID: {target}')
    if target.parent.resolve() == source or source in target.parent.resolve().parents:
        raise D12Error('Corrected provider must live outside the source provider tree')
    calendar = read_provider_calendar(source)
    target.mkdir(parents=True)
    files = _link_tree(source, target, mode=mode)
    from src.d12_price_repair import segment_multipliers

    patches, skipped = [], {}
    instruments = sorted(plan.instrument.unique()) if not plan.empty else []
    for position, instrument in enumerate(instruments, 1):
        if progress:
            progress(f'corrected provider {position}/{len(instruments)}: {instrument}')
        origin = instrument_directory(source, instrument)
        if not origin.is_dir():
            raise D12Error(f'Repair plan names an instrument the provider does not carry: {instrument}')
        cache = {}
        for path in sorted(origin.iterdir()):
            match = BIN_NAME.match(path.name)
            if not match:
                continue
            field = match.group('field')
            if field not in SCALED_FIELDS + INVERSE_FIELDS:
                # Left as a mirrored link. Recorded so an unexpected adjusted field in
                # some other bundle shows up as an inconsistency instead of passing.
                skipped.setdefault(field, []).append(instrument)
                continue
            start, values = read_day_bin(path)
            if start + len(values) > len(calendar):
                raise D12Error(f'{path}: {len(values)} values from offset {start} overrun a '
                               f'{len(calendar)}-session calendar')
            # Fields of one instrument usually share a span; keep the resolved
            # multipliers per (offset, length) so each span is solved once.
            span = (start, len(values))
            if span not in cache:
                cache[span] = segment_multipliers(plan, instrument, calendar[start:start + len(values)]).to_numpy()
            scale = cache[span]
            factor = scale if field in SCALED_FIELDS else 1.0 / scale
            corrected = np.asarray(values, dtype='float64') * factor
            touched = np.isfinite(values) & (scale != 1.0)
            destination = Path(target) / path.relative_to(source)
            destination.unlink()
            write_day_bin(destination, start, corrected)
            written = read_day_bin(destination)[1]
            error = (np.abs(written[touched].astype('float64') / corrected[touched] - 1)
                     if touched.any() else np.zeros(0))
            patches.append({
                'instrument': instrument, 'field': field,
                'file': str(destination.relative_to(target)),
                'calendar_start_index': start, 'values': int(len(values)),
                'values_rescaled': int(touched.sum()),
                'direction': 'scaled' if field in SCALED_FIELDS else 'inverse',
                'minimum_multiplier': float(np.min(scale)), 'maximum_multiplier': float(np.max(scale)),
                'float32_round_trip_error': float(np.max(error)) if error.size else 0.0,
                'sha256': sha256(destination),
            })
    patch_frame = pd.DataFrame(patches, columns=['instrument', 'field', 'file', 'calendar_start_index',
                                                 'values', 'values_rescaled', 'direction',
                                                 'minimum_multiplier', 'maximum_multiplier',
                                                 'float32_round_trip_error', 'sha256'])
    worst = float(patch_frame.float32_round_trip_error.max()) if len(patch_frame) else 0.0
    if worst > 1e-6:
        raise D12Error(f'Corrected bins lose more than float32 precision: {worst:.3e}')
    return patch_frame, {'source_provider': str(source), 'corrected_provider': str(target),
                         'mirror_mode': mode, 'files_mirrored': files,
                         'instruments_patched': len(instruments), 'files_patched': int(len(patch_frame)),
                         'values_rescaled': int(patch_frame.values_rescaled.sum()) if len(patch_frame) else 0,
                         'maximum_float32_round_trip_error': worst,
                         'fields_left_as_links': {field: len(names) for field, names in sorted(skipped.items())},
                         'calendar_sessions': int(len(calendar))}


def verify_corrected_provider(source, target, plan, patches, *, progress=None):
    """Every mirrored file is still the original; every patched file is the intended one.

    Unpatched files are accepted on ``samefile`` (a hard link is the same bytes by
    construction) and fall back to a digest comparison when the mirror was copied.
    Patched files are re-derived from the *source bins and the plan* -- never from the
    file being checked -- so a wrong multiplier, a wrong segment or a wrong calendar
    offset cannot verify itself.
    """
    from src.d12_price_repair import segment_multipliers

    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().resolve()
    patched = {row.file: row for row in patches.itertuples()} if len(patches) else {}
    seen, identical, digested = set(), 0, 0
    for directory in LINKED_DIRECTORIES:
        for path in sorted((source / directory).rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(source))
            mirrored = target / relative
            seen.add(relative)
            if not mirrored.is_file():
                raise D12Error(f'Corrected provider is missing {relative}')
            if relative in patched:
                continue
            if os.path.samefile(path, mirrored):
                identical += 1
            elif sha256(path) == sha256(mirrored):
                digested += 1
            else:
                raise D12Error(f'Mirrored file differs from the original: {relative}')
    extra = {str(p.relative_to(target)) for directory in LINKED_DIRECTORIES
             for p in (target / directory).rglob('*') if p.is_file()} - seen
    if extra:
        raise D12Error('Corrected provider carries files the original does not: ' + ', '.join(sorted(extra)[:10]))
    calendar = read_provider_calendar(source)
    worst = 0.0
    for position, (relative, row) in enumerate(sorted(patched.items()), 1):
        if progress and position % 20 == 0:
            progress(f'verifying patched bins {position}/{len(patched)}')
        if sha256(target / relative) != row.sha256:
            raise D12Error(f'{relative}: patched bin changed after it was written')
        start, original = read_day_bin(source / relative)
        rebuilt_start, rebuilt = read_day_bin(target / relative)
        if start != rebuilt_start or len(original) != len(rebuilt):
            raise D12Error(f'{relative}: patched bin changed shape or calendar offset')
        scale = segment_multipliers(plan, row.instrument, calendar[start:start + len(original)]).to_numpy()
        if row.direction not in {'scaled', 'inverse'}:
            raise D12Error(f'{relative}: unknown patch direction {row.direction!r}')
        expected = np.asarray(original, dtype='float64') * (scale if row.direction == 'scaled' else 1.0 / scale)
        finite = np.isfinite(expected) & np.isfinite(np.asarray(rebuilt, dtype='float64'))
        if finite.sum() != int(np.isfinite(np.asarray(original, dtype='float64')).sum()):
            raise D12Error(f'{relative}: patched bin gained or lost finite values')
        if not finite.any():
            continue
        error = np.abs(np.asarray(rebuilt, dtype='float64')[finite] / expected[finite] - 1)
        worst = max(worst, float(np.max(error)))
    if worst > 1e-6:
        raise D12Error(f'Patched bins do not reproduce the intended values: {worst:.3e}')
    return {'files_checked': len(seen), 'identical_by_inode': identical, 'identical_by_digest': digested,
            'patched_files_reverified': len(patched), 'maximum_reverification_error': worst}

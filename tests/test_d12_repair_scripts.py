"""Stage plumbing for the repair: immutable outputs, stage chaining, and the access override.

These guards are the reason a half-finished repair cannot quietly become a
"repaired R0": a stage refuses to reuse a directory, a later stage refuses to start
on an unfinished or edited earlier stage, and the repaired access object refuses to
serve the registered provider under a corrected name.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repair_io import StageOutput, require_stage, stage_path
from src.d12_repair_access import RepairedDataAccess, guard_process_provider, repaired_config
from src.d12_repo import D12Error


def project(tmp_path):
    """The smallest tree `ResearchDataAccess` and the repair stages actually read."""
    (tmp_path / 'config').mkdir()
    config = {
        'data': {'qlib_provider_uri': str(tmp_path / 'registered'), 'window_marker': 'data/.window_mode.json',
                 'processed_design_dir': 'data/processed/design'},
        'windows': {'design_end': '2018-12-31'},
        'd9': {'v2': {'processed_oos_dir': 'data/processed/oos'}},
    }
    (tmp_path / 'config/config.yaml').write_text(yaml.safe_dump(config))
    for name in ['registered', 'corrected']:
        provider = tmp_path / name
        (provider / 'features').mkdir(parents=True)
        (provider / 'instruments').mkdir(parents=True)
        (provider / 'calendars').mkdir(parents=True)
        (provider / 'calendars/day.txt').write_text('2025-02-17\n')
    return tmp_path, config


# --------------------------------------------------------------------------- stage outputs

def test_a_stage_directory_is_written_once(tmp_path):
    output = StageOutput(tmp_path, 'repair_v1', 'classify')
    output.text('note.txt', 'first')
    with pytest.raises(FileExistsError):
        output.text('note.txt', 'second')
    with pytest.raises(FileExistsError):
        StageOutput(tmp_path, 'repair_v1', 'classify')


def test_stage_filenames_cannot_escape_the_directory(tmp_path):
    output = StageOutput(tmp_path, 'repair_v1', 'classify')
    for name in ['../outside.json', 'sub/dir.json', '..', '.']:
        with pytest.raises(D12Error):
            output.target(name)


def test_repair_and_stage_ids_must_be_simple_names(tmp_path):
    for repair_id, stage in [('../escape', 'classify'), ('repair_v1', 'a/b'), ('', 'classify')]:
        with pytest.raises(D12Error):
            StageOutput(tmp_path, repair_id, stage)


def test_require_stage_reverifies_every_recorded_digest(tmp_path):
    output = StageOutput(tmp_path, 'repair_v1', 'classify')
    output.table('repair_plan', pd.DataFrame({'instrument': ['SZ302132'], 'multiplier': [5.976]}))
    output.finish({'status': 'completed'})
    path, manifest = require_stage(tmp_path, 'repair_v1', 'classify')
    assert manifest['output_sha256']['repair_plan.tsv']
    (path / 'repair_plan.tsv').write_text('tampered\n')
    with pytest.raises(D12Error, match='changed after it completed'):
        require_stage(tmp_path, 'repair_v1', 'classify')


def test_require_stage_refuses_an_unfinished_or_failed_stage(tmp_path):
    with pytest.raises(D12Error, match='has not completed'):
        require_stage(tmp_path, 'repair_v1', 'provider')
    output = StageOutput(tmp_path, 'repair_v1', 'provider')
    output.finish({'status': 'failed', 'error': 'D12Error: no evidence'})
    with pytest.raises(D12Error, match='did not complete'):
        require_stage(tmp_path, 'repair_v1', 'provider')


def test_stage_path_is_addressed_by_repair_and_stage(tmp_path):
    assert stage_path(tmp_path, 'repair_v1', 'r0') == Path(tmp_path).resolve() / 'reports/d12/repair/repair_v1/r0'
    assert stage_path(tmp_path, 'repair_v1', 'run', parent='reports/d12/r0_fixed').name == 'run'


# --------------------------------------------------------------------------- repaired access

def test_repaired_access_points_qlib_at_the_corrected_bundle(tmp_path):
    root, _ = project(tmp_path)
    access = RepairedDataAccess(root, root / 'corrected')
    assert access.provider_uri == str((root / 'corrected').resolve())
    assert access.registered_provider_uri == str(root / 'registered')
    # Every inherited guard survives: an unbounded request is still refused.
    with pytest.raises(Exception):
        access.assert_window('2019-01-01', None)


def test_repaired_access_refuses_the_registered_provider(tmp_path):
    root, _ = project(tmp_path)
    with pytest.raises(D12Error, match='must not point back at the registered provider'):
        RepairedDataAccess(root, root / 'registered')


def test_repaired_access_refuses_an_incomplete_bundle(tmp_path):
    root, _ = project(tmp_path)
    (root / 'corrected/calendars/day.txt').unlink()
    with pytest.raises(D12Error, match='missing calendars/day.txt'):
        RepairedDataAccess(root, root / 'corrected')


def test_process_provider_guard_is_a_no_op_before_qlib_is_configured(tmp_path):
    assert guard_process_provider(tmp_path)['already_initialised'] is False


def test_repaired_config_moves_only_the_two_processed_directories(tmp_path):
    _, config = project(tmp_path)
    config['label'] = {'horizon_trade_days': 5}
    repaired = repaired_config(config, processed_design_dir='data/d12_repair/v1/processed/design',
                               processed_oos_dir='data/d12_repair/v1/processed/oos')
    assert repaired['data']['processed_design_dir'] == 'data/d12_repair/v1/processed/design'
    assert repaired['d9']['v2']['processed_oos_dir'] == 'data/d12_repair/v1/processed/oos'
    assert repaired['label'] == config['label']
    assert repaired['windows'] == config['windows']
    # The original must not be mutated; the registered run is compared against it.
    assert config['data']['processed_design_dir'] == 'data/processed/design'
    assert config['d9']['v2']['processed_oos_dir'] == 'data/processed/oos'


# --------------------------------------------------------------------------- entry point

def test_repair_entry_point_exposes_the_five_stages_in_order():
    import scripts.d12_repair_r0 as entry
    assert entry.STAGES == ('classify', 'provider', 'inputs', 'r0', 'compare')
    assert set(entry.RUNNERS) == set(entry.STAGES)


def test_repair_entry_point_rejects_an_unknown_stage(monkeypatch, capsys):
    import scripts.d12_repair_r0 as entry
    monkeypatch.setattr(sys, 'argv', ['d12_repair_r0.py', '--repair-id', 'v1', '--stage', 'guess'])
    with pytest.raises(SystemExit):
        entry.main()
    assert 'invalid choice' in capsys.readouterr().err


def test_repair_entry_point_rejects_a_coverage_threshold_outside_zero_to_one(monkeypatch, capsys):
    import scripts.d12_repair_r0 as entry
    monkeypatch.setattr(sys, 'argv', ['d12_repair_r0.py', '--repair-id', 'v1', '--stage', 'classify',
                                      '--minimum-coverage', '1.5'])
    with pytest.raises(SystemExit):
        entry.main()
    assert 'minimum-coverage' in capsys.readouterr().err


def test_a_failed_stage_still_leaves_a_manifest_and_a_traceback(tmp_path, monkeypatch):
    import scripts.d12_repair_r0 as entry
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    monkeypatch.setattr(entry, 'load_config', lambda: {})
    monkeypatch.setattr(entry, 'RUNNERS', dict(entry.RUNNERS,
                                               classify=lambda *_: (_ for _ in ()).throw(D12Error('no evidence'))))
    monkeypatch.setattr(sys, 'argv', ['d12_repair_r0.py', '--repair-id', 'v1', '--stage', 'classify'])
    assert entry.main() == 1
    path = stage_path(tmp_path, 'v1', 'classify')
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and 'no evidence' in manifest['error']
    assert 'D12Error' in (path / 'failure.txt').read_text()
    with pytest.raises(D12Error, match='did not complete'):
        require_stage(tmp_path, 'v1', 'classify')


# --------------------------------------------------------------------------- evidence collector

def evidence_project(tmp_path):
    import scripts.d12_collect_factor_evidence as evidence
    config = {
        'data': {'processed_design_dir': 'design', 'tushare_raw_dir': 'raw'},
        'd9': {'v2': {'processed_oos_dir': 'oos'}, 'membership_recovery': {'identifier_alias_removals': [
            {'alias_instrument': 'SZ300114', 'canonical_instrument': 'SZ302132'}]}},
        'universe': {'membership_replacements': [
            {'remove_instrument': 'SH600549', 'add_instrument': 'SH600848'}]},
    }
    for name, rows in [('design', ['SH600000']), ('oos', ['SZ302132', 'SH600000'])]:
        (tmp_path / name).mkdir()
        pd.DataFrame({'trade_date': pd.to_datetime(['2018-06-01'] * len(rows)), 'instrument': rows}
                     ).to_parquet(tmp_path / name / 'benchmark_weights.parquet', index=False)
    return evidence, config


def test_the_evidence_universe_covers_weights_aliases_and_replacements(tmp_path):
    evidence, config = evidence_project(tmp_path)
    names = evidence.universe(config, tmp_path, '2026-08-31')
    assert names == ['SH600000', 'SH600549', 'SH600848', 'SZ300114', 'SZ302132']


def test_the_evidence_universe_refuses_to_guess_when_weights_are_missing(tmp_path):
    evidence, config = evidence_project(tmp_path)
    (tmp_path / 'oos/benchmark_weights.parquet').unlink()
    with pytest.raises(D12Error, match='Missing registered benchmark weights'):
        evidence.universe(config, tmp_path, '2026-08-31')


class FakeTushare:
    """Records every call so a cached security can be shown not to hit the vendor again."""

    def __init__(self, frame):
        self.frame, self.calls = frame, []

    def adj_factor(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame


def test_fetch_normalises_and_refuses_an_invalid_session(tmp_path):
    import scripts.d12_collect_factor_evidence as evidence
    good = pd.DataFrame({'ts_code': ['302132.SZ'], 'trade_date': [20250224], 'adj_factor': [5.976]})
    frame = evidence.fetch_security(FakeTushare(good), '302132.SZ', 'adj_factor', '20250203', '20250307', 0)
    assert frame.trade_date.tolist() == ['20250224']
    bad = pd.DataFrame({'ts_code': ['302132.SZ'], 'trade_date': ['2025-02-24'], 'adj_factor': [5.976]})
    with pytest.raises(D12Error, match='invalid trade_date'):
        evidence.fetch_security(FakeTushare(bad), '302132.SZ', 'adj_factor', '20250203', '20250307', 0)


def test_a_vendor_row_cap_is_treated_as_truncation_not_as_history():
    import scripts.d12_collect_factor_evidence as evidence
    capped = pd.DataFrame({'ts_code': ['302132.SZ'] * 6000,
                           'trade_date': [f'{20000000 + i}' for i in range(101, 6101)],
                           'adj_factor': 1.0})
    with pytest.raises(D12Error, match='assume the call was truncated'):
        evidence.fetch_security(FakeTushare(capped), '302132.SZ', 'adj_factor', '20140101', '20260831', 0)


def test_an_interrupted_pull_resumes_from_its_per_security_cache(tmp_path):
    import scripts.d12_collect_factor_evidence as evidence
    frame = pd.DataFrame({'ts_code': ['302132.SZ'], 'trade_date': [20250224], 'adj_factor': [5.976]})
    fake = FakeTushare(frame)
    cache = tmp_path / 'cache'
    cache.mkdir()
    first = evidence.collect(fake, ['302132.SZ'], 'adj_factor', '20250203', '20250307', cache, 0)
    second = evidence.collect(fake, ['302132.SZ'], 'adj_factor', '20250203', '20250307', cache, 0)
    assert len(fake.calls) == 1, 'the cached security must not be requested again'
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- the gate that repair_v1 lacked

def test_the_separation_gate_rejects_a_threshold_that_cuts_through_noise():
    """`repair_v1` exited 0 with 30,030 segments. This is the check that would not have."""
    from scripts.d12_repair_r0 import separation_gate
    cutting_through_noise = {'one_session_jump_quantiles': {'0.9999': .013},
                             'confirmed_break_jump_minimum': .02}
    with pytest.raises(D12Error, match='cutting through noise'):
        separation_gate(cutting_through_noise, 3.)


def test_the_separation_gate_passes_a_real_step():
    from scripts.d12_repair_r0 import separation_gate
    # The SZ302132 splice against the wobble measured on the same run.
    clean = {'one_session_jump_quantiles': {'0.9999': .013}, 'confirmed_break_jump_minimum': 4.976}
    assert separation_gate(clean, 3.) == pytest.approx(4.976 / .013)


def test_the_separation_gate_stays_quiet_when_there_is_nothing_to_compare():
    from scripts.d12_repair_r0 import separation_gate
    assert separation_gate({'one_session_jump_quantiles': {'0.9999': .01},
                            'confirmed_break_jump_minimum': None}, 3.) is None
    assert separation_gate({'one_session_jump_quantiles': {}, 'confirmed_break_jump_minimum': 1.}, 3.) is None


def test_the_separation_gate_is_wired_into_the_classify_stage():
    import inspect
    import scripts.d12_repair_r0 as entry
    source = inspect.getsource(entry.stage_classify)
    assert 'separation_gate(profile, args.minimum_separation)' in source
    assert "output.json('noise_profile.json', profile)" in source

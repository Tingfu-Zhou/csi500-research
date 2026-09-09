"""Classification, repair planning and the corrected-provider builder.

The regression that matters most is the one built from the recorded evidence:
``SZ302132``'s Qlib factor steps 5.9759x on 2025-02-24 while Tushare's adjustment
factor holds flat at 5.976 across the whole window, and the unadjusted close moves
-1.5198%.  Every number in `test_recorded_sz302132_*` is copied from
``reports/d12/evidence/price_ref_v1/direct_tushare_reference.tsv`` and
``reports/d12/day3/day3_v3/alias_price_focus.tsv``, so if the repair ever stops
turning +488.51% back into -1.52%, these fail.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_price_repair import (ADJUSTED_COLUMNS, apply_factor_repair, build_repair_plan,
                                  classification_summary, classify_factor_rows, repair_effect,
                                  repair_preview, segment_multipliers, verify_repair)
from src.d12_provider import (build_corrected_provider, bin_series, prove_bin_decoding, read_day_bin,
                              read_provider_calendar, verify_corrected_provider, write_day_bin)
from src.d12_repo import D12Error

# The five sessions Qlib served SZ302132 before the step, then the step itself.
SZ302132_RAW = {'2025-02-17': 68.01, '2025-02-18': 64.02, '2025-02-19': 65.90,
                '2025-02-20': 65.50, '2025-02-21': 65.80, '2025-02-24': 64.80,
                '2025-02-25': 63.82}
SZ302132_QLIB_FACTOR = {'2025-02-17': 0.0147037198767066, '2025-02-18': 0.0147037198767066,
                        '2025-02-19': 0.0147037198767066, '2025-02-20': 0.0147037198767066,
                        '2025-02-21': 0.0147037198767066, '2025-02-24': 0.0878683403134346,
                        '2025-02-25': 0.08786981552839279}
TUSHARE_ADJ_FACTOR = 5.976


def audit_frame(raw, factor, reference_factor=None, reference_close=None, instrument='SZ302132'):
    dates = pd.to_datetime(list(raw))
    frame = pd.DataFrame({
        'trade_date': dates, 'instrument': instrument,
        'close': [raw[str(d.date())] * factor[str(d.date())] for d in dates],
        'factor': [factor[str(d.date())] for d in dates],
    })
    frame['raw_close'] = frame.close / frame.factor
    frame['reference_factor'] = [reference_factor.get(str(d.date()), np.nan) for d in dates] \
        if isinstance(reference_factor, dict) else reference_factor
    frame['reference_close'] = [reference_close.get(str(d.date()), np.nan) for d in dates] \
        if isinstance(reference_close, dict) else (
            frame.raw_close if reference_close is None else reference_close)
    frame['mark_return'] = frame.close / frame.close.shift() - 1
    frame['raw_return_1d'] = frame.raw_close / frame.raw_close.shift() - 1
    frame['return_span_sessions'] = 1.0
    frame['limit_status'] = 'valid'
    frame['up_limit'] = frame.raw_close * 1.2
    frame['down_limit'] = frame.raw_close * .8
    frame['weight'] = .0025
    frame['in_project_universe'] = True
    return frame


def recorded_audit():
    return audit_frame(SZ302132_RAW, SZ302132_QLIB_FACTOR, reference_factor=TUSHARE_ADJ_FACTOR)


# --------------------------------------------------------------------------- classification

def test_recorded_sz302132_step_is_classified_as_contradicted():
    classified = classify_factor_rows(recorded_audit())
    step = classified.loc[classified.trade_date.eq('2025-02-24')].iloc[0]
    assert step.verdict == 'factor_contradicted'
    assert step.observed_factor_ratio == pytest.approx(5.975925891558519, rel=1e-9)
    assert step.independent_factor_ratio == pytest.approx(1.0)
    # The original adjusted return, straight off the recorded Qlib quotes.
    assert step.mark_return == pytest.approx(4.8851063486, rel=1e-6)
    assert classified.verdict.eq('factor_consistent').sum() == 5


def test_sessions_without_an_independent_factor_are_never_called_consistent():
    audit = audit_frame(SZ302132_RAW, SZ302132_QLIB_FACTOR, reference_factor=np.nan)
    classified = classify_factor_rows(audit)
    assert set(classified.verdict) == {'no_independent_evidence'}
    assert classification_summary(classified)['independent_coverage'] == 0.0


def test_a_genuine_corporate_action_is_left_alone():
    """Qlib and the vendor agree the factor stepped; that is an ex-rights day, not a defect."""
    raw = {'2024-05-06': 20.0, '2024-05-07': 18.0, '2024-05-08': 18.5}
    factor = {'2024-05-06': 1.0, '2024-05-07': 1.25, '2024-05-08': 1.25}
    reference = {'2024-05-06': 2.0, '2024-05-07': 2.5, '2024-05-08': 2.5}
    classified = classify_factor_rows(audit_frame(raw, factor, reference_factor=reference, instrument='SH600000'))
    assert set(classified.verdict) == {'no_independent_evidence', 'factor_consistent'}
    assert build_repair_plan(classified).empty


def test_ratios_survive_a_suspension_gap():
    """A calendar-shifted ratio goes NaN on the resuming session; observed pairing does not."""
    raw = {'2024-01-02': 10.0, '2024-03-01': 9.0}
    factor = {'2024-01-02': 1.0, '2024-03-01': 3.0}
    reference = {'2024-01-02': 1.0, '2024-03-01': 1.0}
    classified = classify_factor_rows(audit_frame(raw, factor, reference_factor=reference, instrument='SZ000001'))
    resumed = classified.iloc[-1]
    assert resumed.verdict == 'factor_contradicted'
    assert resumed.previous_trade_date == pd.Timestamp('2024-01-02')


def test_a_missing_factor_adjustment_is_caught_in_the_other_direction():
    raw = {'2024-05-06': 20.0, '2024-05-07': 18.0}
    factor = {'2024-05-06': 1.0, '2024-05-07': 1.0}
    reference = {'2024-05-06': 2.0, '2024-05-07': 2.5}
    classified = classify_factor_rows(audit_frame(raw, factor, reference_factor=reference, instrument='SH600001'))
    assert classified.iloc[-1].verdict == 'factor_contradicted'
    plan = build_repair_plan(classified)
    assert len(plan) == 2
    # Default anchor is the earliest segment, so the correction lands on what follows.
    assert float(plan.loc[plan.is_anchor, 'multiplier'].iloc[0]) == 1.0
    assert float(plan.loc[~plan.is_anchor, 'multiplier'].iloc[0]) == pytest.approx(1.25)


# --------------------------------------------------------------------------- planning

def test_recorded_sz302132_repair_restores_the_vendor_return():
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified)
    verify_repair(classified, plan)
    preview = repair_preview(classified, plan)
    step = preview.loc[preview.trade_date.eq('2025-02-24')].iloc[0]
    assert step.original_adjusted_return == pytest.approx(4.8851063486, rel=1e-6)
    # Tushare's own pct_chg for 302132.SZ on 2025-02-24.
    assert step.repaired_adjusted_return == pytest.approx(-0.015197568389057725, rel=1e-6)
    untouched = preview.loc[preview.trade_date.eq('2025-02-25')].iloc[0]
    assert untouched.adjusted_return_change == pytest.approx(0.0, abs=1e-12)


def test_the_anchor_moves_which_rows_change_but_no_return():
    classified = classify_factor_rows(recorded_audit())
    plans = {name: build_repair_plan(classified, anchor=name) for name in ['earliest', 'latest', 'largest']}
    previews = {}
    for name, plan in plans.items():
        verify_repair(classified, plan)
        previews[name] = repair_preview(classified, plan)
    for name in ['latest', 'largest']:
        pd.testing.assert_series_equal(previews['earliest'].repaired_adjusted_return,
                                       previews[name].repaired_adjusted_return, check_names=False, rtol=1e-12)
    assert not plans['earliest'].multiplier.equals(plans['latest'].multiplier)


def test_the_earliest_anchor_leaves_pre_defect_history_at_multiplier_exactly_one():
    """D9 re-asserts the frozen design weights at rtol=1e-12; only an exact 1.0 survives that."""
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified, anchor='earliest')
    before = segment_multipliers(plan, 'SZ302132', pd.to_datetime(
        ['2025-02-17', '2025-02-18', '2025-02-19', '2025-02-20', '2025-02-21']))
    assert (before == 1.0).all()
    values = np.array([68.01, 64.02, 65.90], dtype='float32')
    assert (np.float32(np.float64(values) * 1.0) == values).all()
    assert build_repair_plan(classified, anchor='latest').multiplier.iloc[0] != 1.0


def test_two_breakpoints_chain_multiplicatively():
    raw = {f'2024-01-0{d}': 10.0 for d in range(1, 7)}
    factor = {'2024-01-01': 1.0, '2024-01-02': 1.0, '2024-01-03': 2.0,
              '2024-01-04': 2.0, '2024-01-05': 6.0, '2024-01-06': 6.0}
    classified = classify_factor_rows(audit_frame(raw, factor, reference_factor=1.0, instrument='SZ000002'))
    plan = build_repair_plan(classified, anchor='latest')
    verify_repair(classified, plan)
    multipliers = segment_multipliers(plan, 'SZ000002', pd.to_datetime(list(raw)))
    assert list(multipliers) == pytest.approx([6.0, 6.0, 3.0, 3.0, 1.0, 1.0])


def test_a_raw_price_disagreement_refuses_a_factor_repair():
    audit = recorded_audit()
    audit['reference_close'] = audit.raw_close * 1.5
    classified = classify_factor_rows(audit)
    assert classified.raw_close_contradicted.all()
    with pytest.raises(D12Error, match='raw closes disagree'):
        build_repair_plan(classified)


def test_verify_repair_rejects_a_plan_that_does_not_fix_the_ratio():
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified)
    plan.loc[plan.index[0], 'multiplier'] = 2.0
    with pytest.raises(D12Error, match='still disagrees with evidence'):
        verify_repair(classified, plan)


def test_verify_repair_rejects_a_plan_that_disturbs_a_consistent_ratio():
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified)
    # Interior sessions only, so the breakpoint pair 02-21 -> 02-24 is untouched and
    # the only thing this plan breaks is a ratio that was already correct.
    extra = plan.iloc[[0]].copy()
    extra['segment_start'] = pd.Timestamp('2025-02-18')
    extra['segment_end'] = pd.Timestamp('2025-02-19')
    extra['multiplier'] = float(plan.multiplier.iloc[0]) * 1.5
    with pytest.raises(D12Error, match='disturbed an already-consistent'):
        verify_repair(classified, pd.concat([plan, extra], ignore_index=True))


def test_an_absurd_multiplier_is_refused_rather_than_written():
    raw = {'2024-01-01': 10.0, '2024-01-02': 10.0}
    factor = {'2024-01-01': 1.0, '2024-01-02': 1e9}
    classified = classify_factor_rows(audit_frame(raw, factor, reference_factor=1.0, instrument='SZ000003'))
    with pytest.raises(D12Error, match='implausible factor multiplier'):
        build_repair_plan(classified)


# --------------------------------------------------------------------------- market overlay

def market_frame():
    dates = pd.to_datetime(list(SZ302132_RAW))
    frame = pd.DataFrame({
        'trade_date': dates, 'instrument': 'SZ302132',
        'factor': [SZ302132_QLIB_FACTOR[str(d.date())] for d in dates],
        'volume': 1e6, 'amount': 5e5, 'daily_amount_cny': 5e8,
    })
    for column in ADJUSTED_COLUMNS:
        frame[column] = [SZ302132_RAW[str(d.date())] * SZ302132_QLIB_FACTOR[str(d.date())] for d in dates]
    frame['raw_close'] = frame.close / frame.factor
    return frame


def test_the_market_overlay_leaves_raw_quotes_and_turnover_where_they_were():
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified)
    before = market_frame()
    after = apply_factor_repair(before, plan)
    changed, summary = repair_effect(before, after)
    assert summary['rows_rescaled'] == 2
    assert summary['maximum_raw_close_shift'] == pytest.approx(0.0, abs=1e-12)
    assert summary['maximum_invariant_column_shift']['amount'] == 0.0
    assert summary['maximum_invariant_column_shift']['daily_amount_cny'] == 0.0
    assert (after.loc[after.trade_date.le('2025-02-21'), 'repair_multiplier'] == 1.0).all()
    # Volume is stored divided by the factor, so the raw share count is invariant too.
    raw_shares_before = before.volume * before.factor
    raw_shares_after = after.volume * after.factor
    assert np.allclose(raw_shares_before, raw_shares_after, rtol=1e-12)
    assert len(changed) == 2


def test_an_empty_plan_is_a_no_op_overlay():
    before = market_frame()
    after = apply_factor_repair(before, build_repair_plan(classify_factor_rows(
        audit_frame(SZ302132_RAW, SZ302132_QLIB_FACTOR, reference_factor=np.nan))))
    assert (after.repair_multiplier == 1.0).all()
    pd.testing.assert_frame_equal(before, after.drop(columns='repair_multiplier'))


# --------------------------------------------------------------------------- provider bins

def build_provider(path, calendar, series):
    """A minimal but real Qlib layout: calendars/, instruments/, features/<name>/<field>.day.bin."""
    (path / 'calendars').mkdir(parents=True)
    (path / 'instruments').mkdir(parents=True)
    pd.Series(calendar.strftime('%Y-%m-%d')).to_csv(path / 'calendars/day.txt', index=False, header=False)
    (path / 'instruments/csi500.txt').write_text(
        ''.join(f'{name}\t{calendar[0]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}\n' for name in series))
    for name, fields in series.items():
        directory = path / 'features' / name.lower()
        directory.mkdir(parents=True)
        for field, (start, values) in fields.items():
            write_day_bin(directory / f'{field}.day.bin', start, values)
    return path


def sample_provider(tmp_path):
    calendar = pd.DatetimeIndex(pd.bdate_range('2025-02-10', periods=12))
    dates = [pd.Timestamp(d) for d in SZ302132_RAW]
    start = int(calendar.get_indexer([dates[0]])[0])
    factor = np.array([SZ302132_QLIB_FACTOR[str(d.date())] for d in dates], dtype='float32')
    close = np.array([SZ302132_RAW[str(d.date())] for d in dates], dtype='float32') * factor
    volume = np.full(len(dates), 1e6, dtype='float32')
    amount = np.full(len(dates), 5e5, dtype='float32')
    provider = build_provider(tmp_path / 'source', calendar, {
        'SZ302132': {'close': (start, close), 'factor': (start, factor),
                     'volume': (start, volume), 'amount': (start, amount)},
        'SH600000': {'close': (0, np.arange(len(calendar), dtype='float32') + 10),
                     'factor': (0, np.ones(len(calendar), dtype='float32'))}})
    return provider, calendar, dates


def test_day_bin_round_trips_byte_for_byte(tmp_path):
    values = np.array([1.5, np.nan, 3.25], dtype='float32')
    path = tmp_path / 'close.day.bin'
    write_day_bin(path, 7, values)
    start, read_back = read_day_bin(path)
    assert start == 7
    np.testing.assert_array_equal(np.isnan(read_back), np.isnan(values))
    np.testing.assert_allclose(read_back[~np.isnan(read_back)], values[~np.isnan(values)])
    copy = tmp_path / 'copy.day.bin'
    write_day_bin(copy, start, read_back)
    assert copy.read_bytes() == path.read_bytes()


def test_bin_series_lands_on_the_provider_calendar(tmp_path):
    provider, calendar, dates = sample_provider(tmp_path)
    series = bin_series(provider / 'features/sz302132/factor.day.bin', calendar)
    assert list(series.index) == dates
    assert series.iloc[-1] == pytest.approx(SZ302132_QLIB_FACTOR['2025-02-25'], rel=1e-6)


def test_bin_decoding_proof_fails_loudly_when_the_reader_disagrees(tmp_path):
    provider, calendar, _ = sample_provider(tmp_path)

    def honest(instrument, field, first, last):
        return bin_series(provider / f'features/{instrument.lower()}/{field}.day.bin', calendar).loc[first:last]

    assert prove_bin_decoding(provider, calendar, [('SZ302132', 'factor')], honest)

    def shifted(instrument, field, first, last):
        return honest(instrument, field, first, last) * 1.5

    with pytest.raises(D12Error, match='disagrees with Qlib'):
        prove_bin_decoding(provider, calendar, [('SZ302132', 'factor')], shifted)


def test_corrected_provider_patches_only_the_planned_bins(tmp_path):
    provider, calendar, _ = sample_provider(tmp_path)
    classified = classify_factor_rows(recorded_audit())
    plan = build_repair_plan(classified)
    target = tmp_path / 'corrected'
    patches, summary = build_corrected_provider(provider, target, plan)
    assert summary['instruments_patched'] == 1
    assert set(patches.field) == {'close', 'factor', 'volume'}
    assert set(patches.values_rescaled) == {2}
    # Turnover is not an adjusted quantity, so it stays a link -- and says so.
    assert summary['fields_left_as_links'] == {'amount': 1}
    assert verify_corrected_provider(provider, target, plan, patches)['maximum_reverification_error'] < 1e-6

    # The untouched instrument and the unadjusted turnover series are shared inodes.
    for relative in ['features/sh600000/close.day.bin', 'features/sz302132/amount.day.bin',
                     'calendars/day.txt', 'instruments/csi500.txt']:
        assert (provider / relative).samefile(target / relative)
    assert not (provider / 'features/sz302132/factor.day.bin').samefile(target / 'features/sz302132/factor.day.bin')

    corrected = bin_series(target / 'features/sz302132/factor.day.bin', calendar).dropna()
    original = bin_series(provider / 'features/sz302132/factor.day.bin', calendar).dropna()
    # The step is gone: one flat factor across the whole window.
    assert float(corrected.max() / corrected.min()) == pytest.approx(1.0, abs=2e-5)
    raw_before = bin_series(provider / 'features/sz302132/close.day.bin', calendar).dropna() / original
    raw_after = bin_series(target / 'features/sz302132/close.day.bin', calendar).dropna() / corrected
    np.testing.assert_allclose(raw_after, raw_before, rtol=1e-5)


def test_corrected_provider_verification_catches_a_tampered_bin(tmp_path):
    provider, calendar, _ = sample_provider(tmp_path)
    plan = build_repair_plan(classify_factor_rows(recorded_audit()))
    target = tmp_path / 'corrected'
    patches, _ = build_corrected_provider(provider, target, plan)
    victim = target / 'features/sz302132/factor.day.bin'
    start, values = read_day_bin(victim)
    values[0] *= 1.01
    write_day_bin(victim, start, values)
    with pytest.raises(D12Error, match='changed after it was written'):
        verify_corrected_provider(provider, target, plan, patches)


def test_corrected_provider_verification_catches_a_stray_extra_file(tmp_path):
    provider, calendar, _ = sample_provider(tmp_path)
    plan = build_repair_plan(classify_factor_rows(recorded_audit()))
    target = tmp_path / 'corrected'
    patches, _ = build_corrected_provider(provider, target, plan)
    write_day_bin(target / 'features/sz302132/open.day.bin', 0, np.ones(3, dtype='float32'))
    with pytest.raises(D12Error, match='files the original does not'):
        verify_corrected_provider(provider, target, plan, patches)


def test_corrected_provider_refuses_to_reuse_a_directory(tmp_path):
    provider, _, _ = sample_provider(tmp_path)
    plan = build_repair_plan(classify_factor_rows(recorded_audit()))
    target = tmp_path / 'corrected'
    build_corrected_provider(provider, target, plan)
    with pytest.raises(D12Error, match='already exists'):
        build_corrected_provider(provider, target, plan)


def test_provider_calendar_is_read_from_the_source_bundle(tmp_path):
    provider, calendar, _ = sample_provider(tmp_path)
    assert list(read_provider_calendar(provider)) == list(calendar)


# --------------------------------------------------------------------------- evidence caches

def reference_rows(factor):
    return pd.DataFrame({'trade_date': pd.to_datetime(['2025-02-24']), 'instrument': ['SZ302132'],
                         'reference_factor': [factor]})


def test_the_same_observation_in_two_caches_is_one_observation():
    from src.d12_price_repair import resolve_reference_duplicates
    both = pd.concat([reference_rows(5.976), reference_rows(5.976)], ignore_index=True)
    resolved = resolve_reference_duplicates(both, 'reference_factor')
    assert len(resolved) == 1 and float(resolved.reference_factor.iloc[0]) == 5.976


def test_two_caches_that_actually_disagree_stop_the_repair():
    from src.d12_price_repair import resolve_reference_duplicates
    both = pd.concat([reference_rows(5.976), reference_rows(1.0)], ignore_index=True)
    with pytest.raises(D12Error, match='caches disagree'):
        resolve_reference_duplicates(both, 'reference_factor')


def test_an_empty_plan_still_produces_a_usable_preview_and_design_check():
    """No confirmed defect anywhere is a legitimate outcome; the classify stage must survive it."""
    classified = classify_factor_rows(audit_frame(SZ302132_RAW, SZ302132_QLIB_FACTOR, reference_factor=np.nan))
    plan = build_repair_plan(classified)
    assert plan.empty
    preview = repair_preview(classified, plan)
    assert preview.empty
    # Both columns the classify stage filters and reports on must exist even when empty.
    assert preview.adjusted_return_change.abs().gt(1e-9).sum() == 0
    assert preview.verdict.eq('factor_contradicted').sum() == 0
    assert plan.loc[plan.breakpoint_date.notna() & plan.breakpoint_date.le(pd.Timestamp('2018-12-31'))].empty
    assert verify_repair(classified, plan)['checked_ratios'] == 0

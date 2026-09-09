"""Classify Qlib factor-chain defects against independent evidence, and plan repairs.

Scope, stated once so no caller has to infer it.

This module never decides that a large adjusted return is wrong.  A genuine
ex-rights day produces exactly the same screen hit as a broken splice.  The only
thing that separates them is an *independent* adjustment factor for the same
security on the same two sessions, and the only verdict this module reaches
without one is ``no_independent_evidence``.

The repair it plans is deliberately the smallest one that can fix a splice: a
piecewise-constant multiplier on the Qlib factor, changing only at the sessions
where the Qlib factor ratio contradicts the independent ratio.  Raw prices --
``close / factor`` -- are invariant under it by construction, which is the point:
the raw quotes already reconcile with the vendor to ~1e-8, so a repair that moved
them would be repairing the wrong thing.

What it will refuse to do: rescale an instrument whose *raw* closes disagree with
the independent source.  That is a different defect with a different fix, and a
multiplier cannot express it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.d12_repo import D12Error

KEYS = ['trade_date', 'instrument']
#: Same thresholds `src.d12_day3.reference_crosscheck` already screens with, so a
#: row this module calls contradicted is a row day-3 already flagged.
FACTOR_TOL = 1e-3
RAW_TOL = 1e-4
#: Adjusted price columns move with the factor; volume is stored divided by it;
#: turnover in CNY is unadjusted and must not move at all.
ADJUSTED_COLUMNS = ('open', 'high', 'low', 'close', 'vwap')
INVERSE_COLUMNS = ('volume',)
INVARIANT_COLUMNS = ('amount', 'daily_amount_cny', 'up_limit', 'down_limit')

VERDICTS = ('factor_consistent', 'factor_contradicted', 'no_independent_evidence')


def resolve_reference_duplicates(frame, column, *, rtol=1e-12):
    """Collapse the same observation seen in two evidence caches; keep real conflicts.

    The narrow ``price_ref_v1`` probe and a later full-period pull both cover
    2025-02, so the same ``(session, security)`` arrives twice.  Two rows carrying
    the same number are one observation and must not stop a repair; two rows
    carrying *different* numbers are a genuine disagreement between caches and must.
    """
    frame = frame.dropna(subset=[column]).copy()
    if frame.empty:
        return frame
    grouped = frame.groupby(KEYS, sort=False)[column]
    spread = (grouped.transform('max') - grouped.transform('min')).abs()
    scale = grouped.transform('max').abs().clip(lower=1e-12)
    conflicting = spread.gt(scale * rtol)
    if conflicting.any():
        offenders = frame.loc[conflicting, KEYS].drop_duplicates().head(5).to_dict('records')
        raise D12Error(f'Independent {column} caches disagree on {int(conflicting.sum())} row(s): {offenders}')
    return frame.drop_duplicates(KEYS).reset_index(drop=True)


def _observed(frame, column):
    values = pd.to_numeric(frame[column], errors='coerce')
    return values.where(np.isfinite(values) & values.gt(0))


def classify_factor_rows(audit, *, factor_tol=FACTOR_TOL, raw_tol=RAW_TOL):
    """Per-session verdict on the Qlib factor chain, on *observed* sessions only.

    Ratios are rebuilt here rather than reused from ``price_audit``.  That routine
    reindexes onto the exchange calendar and shifts by one calendar row, so a
    suspension leaves a NaN ratio on the session that resumes -- exactly where a
    bad splice is most likely to hide.  Pairing consecutive observed sessions
    instead keeps those comparisons alive, and records which prior session each
    ratio was taken against so a reviewer can check the pairing by hand.

    A verdict needs the independent factor on *both* sessions of the pair.  With
    one of them missing the row is ``no_independent_evidence``; it is never
    silently treated as agreement.
    """
    for column in KEYS + ['factor']:
        if column not in audit:
            raise D12Error(f'Factor classification needs column {column!r}')
    frame = audit.loc[_observed(audit, 'factor').notna()].copy()
    if frame.duplicated(KEYS).any():
        raise D12Error('Duplicate instrument/date rows in the price audit')
    frame['trade_date'] = pd.to_datetime(frame.trade_date).dt.normalize()
    frame = frame.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
    if 'reference_factor' not in frame:
        frame['reference_factor'] = np.nan
    if 'reference_close' not in frame:
        frame['reference_close'] = np.nan
    factor, reference = _observed(frame, 'factor'), _observed(frame, 'reference_factor')
    frame['factor'] = factor
    frame['reference_factor'] = reference
    frame['previous_trade_date'] = frame.trade_date.groupby(frame.instrument).shift()
    frame['previous_factor'] = factor.groupby(frame.instrument).shift()
    frame['previous_reference_factor'] = reference.groupby(frame.instrument).shift()
    frame['observed_factor_ratio'] = frame.factor / frame.previous_factor
    frame['independent_factor_ratio'] = frame.reference_factor / frame.previous_reference_factor
    frame['factor_ratio_error'] = (frame.observed_factor_ratio / frame.independent_factor_ratio - 1).abs()
    if 'raw_close' in frame:
        raw = _observed(frame, 'raw_close')
    else:
        raw = frame.close / frame.factor if 'close' in frame else pd.Series(np.nan, index=frame.index)
    frame['raw_close_error'] = (raw / _observed(frame, 'reference_close') - 1).abs()
    # A disagreeing raw quote is a separate defect; a factor multiplier cannot fix it.
    frame['raw_close_contradicted'] = frame.raw_close_error.gt(raw_tol).fillna(False)
    comparable = frame.observed_factor_ratio.notna() & frame.independent_factor_ratio.notna()
    frame['verdict'] = np.select(
        [~comparable, frame.factor_ratio_error.le(factor_tol)],
        ['no_independent_evidence', 'factor_consistent'], default='factor_contradicted')
    # The repair multiplier that makes this session's ratio match the evidence.
    frame['implied_correction'] = (frame.independent_factor_ratio / frame.observed_factor_ratio).where(
        frame.verdict.eq('factor_contradicted'))
    return frame


def classification_summary(classified):
    counts = classified.verdict.value_counts()
    contradicted = classified.loc[classified.verdict.eq('factor_contradicted')]
    return {
        'observed_rows': int(len(classified)),
        'instruments': int(classified.instrument.nunique()),
        **{verdict: int(counts.get(verdict, 0)) for verdict in VERDICTS},
        'independent_coverage': float(classified.verdict.ne('no_independent_evidence').mean())
        if len(classified) else 0.0,
        'contradicted_instruments': int(contradicted.instrument.nunique()),
        'raw_close_contradicted_rows': int(classified.raw_close_contradicted.sum()),
        'raw_close_contradicted_instruments': int(
            classified.loc[classified.raw_close_contradicted, 'instrument'].nunique()),
    }


def build_repair_plan(classified, *, anchor='earliest', max_multiplier=1e6):
    """Piecewise-constant factor multipliers, one segment per contradicted session.

    Each contradicted session ``d`` is a breakpoint: everything before ``d`` sits on
    a different scale from everything at or after it.  One segment is the anchor and
    keeps multiplier 1; the rest are derived from it so that every corrected ratio
    reproduces the independent ratio exactly.

    Only ratios matter downstream -- features, labels, benchmark weights and marks all
    divide one price by another, and an order's notional is units times price -- so
    the anchor cannot change a single return.  What it changes is *which* rows get
    rewritten, and that does matter here.

    ``anchor='earliest'`` is the default because it leaves every session before the
    first contradiction with multiplier exactly 1.0, which round-trips through float32
    bit-for-bit.  D9 re-asserts the frozen design-window weights at ``rtol=1e-12``, so
    a plan that rescaled 2014-2018 quotes -- even by a value that cannot move a
    return -- would fail that assertion on float32 re-rounding alone.  ``'latest'``
    (today's prices stay put) and ``'largest'`` (fewest rows rewritten) remain
    available for cases where the design window is not in the span.
    """
    if anchor not in {'earliest', 'latest', 'largest'}:
        raise D12Error("anchor must be 'earliest', 'latest' or 'largest'")
    rows = []
    for instrument, group in classified.groupby('instrument', sort=True):
        group = group.sort_values('trade_date')
        breaks = group.loc[group.verdict.eq('factor_contradicted')]
        if breaks.empty:
            continue
        if group.raw_close_contradicted.any():
            raise D12Error(
                f'{instrument}: raw closes disagree with the independent source on '
                f'{int(group.raw_close_contradicted.sum())} session(s); a factor multiplier '
                'cannot express that defect. Exclude the instrument or resolve the raw quotes first.')
        edges = list(breaks.trade_date)
        starts = [group.trade_date.iloc[0]] + edges
        ends = [date - pd.Timedelta(days=1) for date in edges] + [group.trade_date.iloc[-1]]
        counts = [int(group.trade_date.between(lo, hi).sum()) for lo, hi in zip(starts, ends)]
        index = {'earliest': 0, 'latest': len(starts) - 1}.get(anchor, int(np.argmax(counts)))
        multipliers = [1.0] * len(starts)
        # Walk outward from the anchor; each step consumes exactly one breakpoint.
        for position in range(index - 1, -1, -1):
            correction = float(breaks.implied_correction.iloc[position])
            multipliers[position] = multipliers[position + 1] / correction
        for position in range(index + 1, len(starts)):
            correction = float(breaks.implied_correction.iloc[position - 1])
            multipliers[position] = multipliers[position - 1] * correction
        for position, (lo, hi, multiplier, count) in enumerate(zip(starts, ends, multipliers, counts)):
            if not np.isfinite(multiplier) or multiplier <= 0 or multiplier > max_multiplier or multiplier < 1 / max_multiplier:
                raise D12Error(f'{instrument}: implausible factor multiplier {multiplier!r} on segment starting {lo.date()}')
            rows.append({
                'instrument': instrument, 'segment_start': lo, 'segment_end': hi,
                'multiplier': multiplier, 'observed_rows': count,
                'is_anchor': position == index, 'anchor_rule': anchor,
                'breakpoint_date': edges[position] if position < len(edges) else pd.NaT,
                'breakpoint_observed_ratio': float(breaks.observed_factor_ratio.iloc[position])
                if position < len(edges) else np.nan,
                'breakpoint_independent_ratio': float(breaks.independent_factor_ratio.iloc[position])
                if position < len(edges) else np.nan,
                'breakpoint_previous_session': breaks.previous_trade_date.iloc[position]
                if position < len(edges) else pd.NaT,
            })
    plan = pd.DataFrame(rows, columns=['instrument', 'segment_start', 'segment_end', 'multiplier',
                                       'observed_rows', 'is_anchor', 'anchor_rule', 'breakpoint_date',
                                       'breakpoint_observed_ratio', 'breakpoint_independent_ratio',
                                       'breakpoint_previous_session'])
    if not plan.empty:
        for column in ['segment_start', 'segment_end', 'breakpoint_date', 'breakpoint_previous_session']:
            plan[column] = pd.to_datetime(plan[column])
    return plan


def segment_multipliers(plan, instrument, dates):
    """Multiplier for each date of one instrument; 1.0 outside every planned segment."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    result = pd.Series(1.0, index=dates)
    segments = plan.loc[plan.instrument.eq(instrument)]
    for row in segments.itertuples():
        result.loc[(dates >= row.segment_start) & (dates <= row.segment_end)] = float(row.multiplier)
    return result


def apply_factor_repair(market, plan):
    """Rescale the planned segments in a loaded market frame.

    Adjusted quotes and the factor move together and volume moves inversely, so the
    raw quote ``close / factor`` and the raw share count ``volume * factor`` both
    come out bit-comparable to what they were.  Turnover in CNY and the exchange
    limits are untouched: neither is expressed in adjusted units.
    """
    result = market.copy()
    if plan.empty:
        result['repair_multiplier'] = 1.0
        return result
    result['trade_date'] = pd.to_datetime(result.trade_date).dt.normalize()
    multiplier = pd.Series(1.0, index=result.index)
    for instrument, segments in plan.groupby('instrument', sort=False):
        rows = result.instrument.eq(instrument)
        if not rows.any():
            continue
        for row in segments.itertuples():
            inside = rows & result.trade_date.between(row.segment_start, row.segment_end)
            multiplier.loc[inside] = float(row.multiplier)
    for column in ADJUSTED_COLUMNS + ('factor',):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors='coerce') * multiplier
    for column in INVERSE_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors='coerce') / multiplier
    for column in ['raw_high', 'raw_low', 'raw_close']:
        if column in result and column[4:] in result:
            result[column] = result[column[4:]] / result.factor
    result['repair_multiplier'] = multiplier
    return result


def verify_repair(classified, plan, *, factor_tol=FACTOR_TOL):
    """Prove the plan on the classified rows before any file is written.

    Two properties, both checked on the real data rather than assumed: every
    contradicted ratio now reproduces the independent ratio, and no ratio that was
    already consistent was disturbed.
    """
    if plan.empty:
        contradicted = int(classified.verdict.eq('factor_contradicted').sum())
        if contradicted:
            raise D12Error(f'{contradicted} contradicted session(s) but an empty repair plan')
        return {'checked_instruments': 0, 'checked_ratios': 0,
                'maximum_corrected_ratio_error': 0.0, 'maximum_consistent_ratio_shift': 0.0}
    worst_corrected = worst_consistent = 0.0
    checked = 0
    for instrument, group in classified.loc[classified.instrument.isin(plan.instrument)].groupby('instrument'):
        group = group.sort_values('trade_date')
        multiplier = segment_multipliers(plan, instrument, group.trade_date)
        scaled = group.factor.to_numpy() * multiplier.to_numpy()
        previous = np.concatenate([[np.nan], scaled[:-1]])
        corrected_ratio = scaled / previous
        contradicted = group.verdict.eq('factor_contradicted').to_numpy()
        consistent = group.verdict.eq('factor_consistent').to_numpy()
        independent = group.independent_factor_ratio.to_numpy()
        observed = group.observed_factor_ratio.to_numpy()
        if contradicted.any():
            error = np.abs(corrected_ratio[contradicted] / independent[contradicted] - 1)
            worst_corrected = max(worst_corrected, float(np.nanmax(error)))
            checked += int(contradicted.sum())
        if consistent.any():
            shift = np.abs(corrected_ratio[consistent] / observed[consistent] - 1)
            worst_consistent = max(worst_consistent, float(np.nanmax(shift)))
    if worst_corrected > factor_tol:
        raise D12Error(f'Repaired factor ratio still disagrees with evidence: max error {worst_corrected:.3e}')
    if worst_consistent > factor_tol:
        raise D12Error(f'Repair disturbed an already-consistent factor ratio: max shift {worst_consistent:.3e}')
    return {'checked_instruments': int(plan.instrument.nunique()), 'checked_ratios': checked,
            'maximum_corrected_ratio_error': worst_corrected,
            'maximum_consistent_ratio_shift': worst_consistent}


def repair_preview(classified, plan):
    """Before/after adjusted return for every session of every instrument in the plan.

    Built from the raw quote and the factor alone, so it can be checked by hand
    against the vendor's own ``pct_chg`` before a single provider file is written.
    """
    columns = ['repair_multiplier', 'repaired_factor', 'original_adjusted_return',
               'repaired_adjusted_return', 'adjusted_return_change']
    if plan.empty:
        return classified.iloc[:0].assign(**{name: pd.Series(dtype=float) for name in columns})
    rows = []
    for instrument, group in classified.loc[classified.instrument.isin(plan.instrument)].groupby('instrument'):
        group = group.sort_values('trade_date').copy()
        group['repair_multiplier'] = segment_multipliers(plan, instrument, group.trade_date).to_numpy()
        group['repaired_factor'] = group.factor * group.repair_multiplier
        raw = group.raw_close if 'raw_close' in group else group.close / group.factor
        group['original_adjusted_return'] = (raw * group.factor) / (raw * group.factor).shift() - 1
        group['repaired_adjusted_return'] = (raw * group.repaired_factor) / (raw * group.repaired_factor).shift() - 1
        rows.append(group)
    preview = pd.concat(rows, ignore_index=True)
    preview['adjusted_return_change'] = preview.repaired_adjusted_return - preview.original_adjusted_return
    return preview


def repair_effect(before, after, *, keys=KEYS):
    """Row-level before/after for the columns a repair is allowed to move."""
    columns = [c for c in ADJUSTED_COLUMNS + ('factor',) + INVERSE_COLUMNS if c in before and c in after]
    left = before[keys + columns].rename(columns={c: 'before_' + c for c in columns})
    right = after[keys + columns + ['repair_multiplier']].rename(columns={c: 'after_' + c for c in columns})
    merged = left.merge(right, on=keys, how='inner', validate='one_to_one')
    changed = merged.repair_multiplier.ne(1.0)
    invariants = {}
    for column in INVARIANT_COLUMNS:
        if column in before and column in after:
            joined = before[keys + [column]].merge(after[keys + [column]], on=keys, suffixes=('_before', '_after'))
            difference = (joined[column + '_after'] - joined[column + '_before']).abs()
            invariants[column] = float(np.nanmax(difference)) if len(difference) else 0.0
    raw_shift = 0.0
    if 'close' in columns and 'factor' in columns:
        raw_before = merged.before_close / merged.before_factor
        raw_after = merged.after_close / merged.after_factor
        raw_shift = float(np.nanmax((raw_after / raw_before - 1).abs())) if len(merged) else 0.0
    return merged.loc[changed].copy(), {
        'rows_compared': int(len(merged)), 'rows_rescaled': int(changed.sum()),
        'instruments_rescaled': int(merged.loc[changed, 'instrument'].nunique()),
        'maximum_raw_close_shift': raw_shift,
        'maximum_invariant_column_shift': invariants,
    }

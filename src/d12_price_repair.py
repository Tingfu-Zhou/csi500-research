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
#: One-session jump that makes a session worth *looking at*. Deliberately a screen,
#: not a verdict: on the real bundle the community `$factor` wobbles up to about
#: five parts in a thousand around its level (see `classify_factor_rows`), so a
#: number in this range separates candidates from noise but proves nothing by itself.
SCREEN_TOL = 5e-3
#: How far the robust level either side of a candidate must actually move before the
#: candidate is called a break. Same scale as the screen: what promotes a candidate is
#: that the shift *survives medians*, not that it is larger.
CONFIRM_TOL = 5e-3
#: Observed sessions per side of the median comparison. Small enough that a five-session
#: segment (SZ302132 before its splice) still gets a full window.
LEVEL_WINDOW = 5
#: Observations a side must actually have before its median means anything. A session
#: sitting next to an already-confirmed break gets a one-observation window, and a
#: "median" of one is the observation itself -- enough to invent a second break out of
#: ordinary wobble. Below this a candidate is left unconfirmed, which is the honest
#: answer: two adjacent splices are indistinguishable from one splice plus noise.
MIN_LEVEL_OBSERVATIONS = 3
RAW_TOL = 1e-4
#: Above this share of a security's sessions being breaks, a piecewise-constant model
#: is not describing the data and the plan is refused rather than written. Only
#: applied once a series is long enough for a share to mean anything -- a two-session
#: fixture with one splice is 50% breaks and perfectly well described.
MAX_BREAK_SHARE = .02
MIN_SESSIONS_FOR_SHARE_GUARD = 50
#: Adjusted price columns move with the factor; volume is stored divided by it;
#: turnover in CNY is unadjusted and must not move at all.
ADJUSTED_COLUMNS = ('open', 'high', 'low', 'close', 'vwap')
INVERSE_COLUMNS = ('volume',)
INVARIANT_COLUMNS = ('amount', 'daily_amount_cny', 'up_limit', 'down_limit')

VERDICTS = ('factor_consistent', 'factor_contradicted', 'factor_noise_not_persistent',
            'no_independent_evidence')


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


def _level_shift(values, position, window, boundaries, minimum):
    """Median of ``scale`` after vs before ``position``, clipped at ``boundaries``.

    ``None`` when a side was cut short *by a confirmed break* and has fewer than
    ``minimum`` observations left: there, more data exists and we chose not to use it,
    so a one-observation "median" would be reading noise as a level.  A side limited
    by the start or end of the series is different -- that is all the evidence there
    is, and a five-session history with a six-fold step in it is not ambiguous.
    """
    below = [b for b in boundaries if b <= position]
    above = [b for b in boundaries if b > position]
    low, high = max(below, default=0), min(above, default=len(values))
    before = values[max(low, position - window):position]
    after = values[position:min(high, position + window)]
    before, after = before[np.isfinite(before)], after[np.isfinite(after)]
    if before.size == 0 or after.size == 0:
        return None
    if (below and before.size < minimum) or (above and after.size < minimum):
        return None
    return float(np.median(after) / np.median(before))


def _confirm_breakpoints(scale, candidates, strength, *, window, confirm_tol,
                         minimum=MIN_LEVEL_OBSERVATIONS):
    """Keep only candidates where the *level* of ``scale`` shifts and stays shifted.

    ``scale`` is one instrument's series, positionally indexed, and ``strength`` is
    each session's own one-session jump on a log scale.  A candidate is judged on
    robust levels -- medians of up to ``window`` observed sessions on either side --
    because that is what separates a splice from the bundle's wobble.

    Two details here are load-bearing, and both were wrong in the obvious version.

    A window is clipped only at breaks already *confirmed*, never at other candidates.
    An isolated one-session outlier is itself a candidate, so clipping at candidates
    shortens its neighbour's window to one or two values -- and a median of two is
    just their midpoint, which the outlier then drags far enough to confirm a break
    that is not there.

    And candidates are settled by descending *jump*, not by descending level shift.
    The session after a real step sees the same level shift as the step itself, so
    ranking on level shift can settle the wrong one of the pair and leave the actual
    step to be confirmed separately against it -- two breaks for one splice.  The
    jump is unambiguous: it is large exactly at the session where the factor moved.
    """
    values = np.asarray(scale, dtype='float64')
    strength = np.asarray(strength, dtype='float64')
    confirmed, levels = [], {}
    for position in sorted(candidates, key=lambda p: -strength[p]):
        shift = _level_shift(values, position, window, confirmed, minimum)
        if shift is None:
            continue
        levels[position] = shift
        if abs(shift - 1) > confirm_tol:
            confirmed.append(position)
            confirmed.sort()
    return confirmed, levels


def classify_factor_rows(audit, *, screen_tol=SCREEN_TOL, confirm_tol=CONFIRM_TOL,
                         window=LEVEL_WINDOW, minimum=MIN_LEVEL_OBSERVATIONS, raw_tol=RAW_TOL):
    """Per-session verdict on the Qlib factor chain, on *observed* sessions only.

    The quantity that matters is the *level*

        scale(t) = factor(t) / independent_factor(t)

    which is constant for as long as the two adjustment chains agree, whatever
    per-instrument normalisation each vendor uses.  A splice moves that level and
    leaves it moved; that is the only defect a piecewise-constant repair can express.

    Testing a single session's factor ratio instead is what the first version of
    this module did, and on the real bundle it was wrong: the community `$factor`
    wobbles around its level by one to five parts in a thousand and *reverts*
    (SH600017 returns to 0.488968402 on three separate flagged dates; SH600021 steps
    down 0.11% and back up six weeks later).  At a 1e-3 ratio tolerance that produced
    29,529 "contradictions" over 501 securities -- 98% of them between 0.1% and 0.5%,
    piled against the threshold, which is the shape of a cut through noise rather
    than a set of findings.  So a candidate is screened on the one-session jump and
    then has to survive a comparison of medians on both sides; noise cancels there,
    a splice does not.

    Ratios are rebuilt here rather than reused from ``price_audit``: that routine
    reindexes onto the exchange calendar and shifts by one calendar row, so a
    suspension leaves a NaN ratio on the session that resumes -- exactly where a bad
    splice is most likely to hide.  Pairing consecutive observed sessions keeps those
    comparisons alive and records which prior session each ratio was taken against.

    A verdict needs the independent factor on *both* sessions of the pair.  With one
    missing the row is ``no_independent_evidence``; never silently agreement.
    """
    for column in KEYS + ['factor']:
        if column not in audit:
            raise D12Error(f'Factor classification needs column {column!r}')
    if not 0 < screen_tol <= confirm_tol:
        raise D12Error('screen_tol must be positive and no larger than confirm_tol')
    frame = audit.loc[_observed(audit, 'factor').notna()].copy()
    if frame.duplicated(KEYS).any():
        raise D12Error('Duplicate instrument/date rows in the price audit')
    frame['trade_date'] = pd.to_datetime(frame.trade_date).dt.normalize()
    frame = frame.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
    for column in ['reference_factor', 'reference_close']:
        if column not in frame:
            frame[column] = np.nan
    factor, reference = _observed(frame, 'factor'), _observed(frame, 'reference_factor')
    frame['factor'] = factor
    frame['reference_factor'] = reference
    frame['previous_trade_date'] = frame.trade_date.groupby(frame.instrument).shift()
    frame['previous_factor'] = factor.groupby(frame.instrument).shift()
    frame['previous_reference_factor'] = reference.groupby(frame.instrument).shift()
    frame['observed_factor_ratio'] = frame.factor / frame.previous_factor
    frame['independent_factor_ratio'] = frame.reference_factor / frame.previous_reference_factor
    # The chain-alignment level, and its one-session jump. `scale_jump` and
    # `factor_ratio_error` are the same number; the first name says what it measures.
    frame['scale'] = frame.factor / frame.reference_factor
    frame['scale_jump'] = (frame.observed_factor_ratio / frame.independent_factor_ratio - 1)
    frame['factor_ratio_error'] = frame.scale_jump.abs()
    # Log scale so a halving and a doubling rank as equally strong evidence.
    frame['jump_strength'] = np.abs(np.log((1 + frame.scale_jump).where(lambda x: x.gt(0)))).fillna(0.0)
    if 'raw_close' in frame:
        raw = _observed(frame, 'raw_close')
    else:
        raw = frame.close / frame.factor if 'close' in frame else pd.Series(np.nan, index=frame.index)
    frame['raw_close_error'] = (raw / _observed(frame, 'reference_close') - 1).abs()
    # A disagreeing raw quote is a separate defect; a factor multiplier cannot fix it.
    frame['raw_close_contradicted'] = frame.raw_close_error.gt(raw_tol).fillna(False)
    comparable = frame.observed_factor_ratio.notna() & frame.independent_factor_ratio.notna()

    frame['segment_index'] = 0
    frame['level_shift'] = np.nan
    screened = comparable & frame.factor_ratio_error.gt(screen_tol)
    verdict = np.where(comparable, 'factor_consistent', 'no_independent_evidence').astype(object)
    for _, positions in frame.groupby('instrument', sort=False).indices.items():
        order = np.sort(positions)
        candidates = [int(i) for i, position in enumerate(order) if screened.iat[position]]
        if not candidates:
            continue
        confirmed, shifts = _confirm_breakpoints(
            frame.scale.to_numpy()[order], candidates, frame.jump_strength.to_numpy()[order],
            window=window, confirm_tol=confirm_tol, minimum=minimum)
        for local, shift in shifts.items():
            frame.iat[order[local], frame.columns.get_loc('level_shift')] = shift
        confirmed_set = set(confirmed)
        for local in candidates:
            verdict[order[local]] = ('factor_contradicted' if local in confirmed_set
                                     else 'factor_noise_not_persistent')
        if confirmed:
            marks = np.zeros(len(order), dtype=int)
            marks[np.asarray(confirmed, dtype=int)] = 1
            frame.iloc[order, frame.columns.get_loc('segment_index')] = marks.cumsum()
    frame['verdict'] = verdict
    # Detection and correction are different problems and use different estimators.
    # `level_shift` (medians) is what *detects* a break, because it has to survive
    # the wobble. The correction is the exact ratio that puts this session's adjusted
    # return back on the independent one -- that target is known, so estimating it
    # robustly would only add error to the single return the multiplier can move.
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
        'screened_but_not_persistent_instruments': int(
            classified.loc[classified.verdict.eq('factor_noise_not_persistent'), 'instrument'].nunique()),
        'raw_close_contradicted_rows': int(classified.raw_close_contradicted.sum()),
        'raw_close_contradicted_instruments': int(
            classified.loc[classified.raw_close_contradicted, 'instrument'].nunique()),
    }


def noise_profile(classified, *, quantiles=(.5, .9, .99, .999, .9999, 1.0)):
    """The measured one-session wobble on everything not confirmed as a break.

    Reported next to the thresholds so the cut is auditable against the data that
    run actually saw, instead of being asserted from a previous run's numbers.
    """
    kept = classified.loc[classified.verdict.isin(['factor_consistent', 'factor_noise_not_persistent'])]
    jump = kept.factor_ratio_error.dropna()
    breaks = classified.loc[classified.verdict.eq('factor_contradicted'), 'factor_ratio_error'].dropna()
    return {
        'sessions_measured': int(len(jump)),
        'one_session_jump_quantiles': {str(q): float(jump.quantile(q)) for q in quantiles} if len(jump) else {},
        'confirmed_break_jump_minimum': float(breaks.min()) if len(breaks) else None,
        'confirmed_breaks': int(len(breaks)),
        'note': 'A healthy separation shows the unconfirmed wobble ending well below the '
                'smallest confirmed break. Overlap means the thresholds are cutting through noise.',
    }


def build_repair_plan(classified, *, anchor='earliest', max_multiplier=1e6,
                      max_break_share=MAX_BREAK_SHARE,
                      min_sessions_for_share_guard=MIN_SESSIONS_FOR_SHARE_GUARD):
    """Piecewise-constant factor multipliers, one segment per confirmed break.

    Each confirmed session ``d`` is a breakpoint: everything before ``d`` sits on a
    different chain level from everything at or after it.  One segment is the anchor
    and keeps multiplier 1; the rest are derived from it so that the corrected level
    is the same on both sides of every break.

    ``max_break_share`` is a model-fit guard, not a tuning knob.  A security whose
    sessions are more than a couple of percent breakpoints is not a spliced series
    with a few steps in it -- it is a series this model does not describe -- and the
    plan is refused instead of written.

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
        share = len(breaks) / max(1, len(group))
        if len(group) >= min_sessions_for_share_guard and share > max_break_share:
            raise D12Error(
                f'{instrument}: {len(breaks)} of {len(group)} observed sessions ({share:.1%}) came back as '
                f'breaks, above the {max_break_share:.1%} model-fit limit. A piecewise-constant multiplier '
                'is not describing this series; review the classification before planning a repair.')
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


def verify_repair(classified, plan, *, ratio_tol=1e-12, level_tol=CONFIRM_TOL, window=LEVEL_WINDOW,
                  minimum=MIN_LEVEL_OBSERVATIONS):
    """Prove the plan on the classified rows before any file is written.

    Three properties, all measured on the real data rather than assumed.

    Every confirmed break now reproduces the independent ratio exactly -- the target
    is known, so this one is checked at floating-point scale.  Nothing inside a
    segment moved: a constant multiplier cancels in a ratio, so within-segment ratios
    must come back bit-for-bit.  And the residual level mismatch across each break is
    *reported*, not gated: the correction is anchored on one session's factors, and
    those carry the provider's own wobble, so a residual of a few parts in a thousand
    is the data's precision rather than a defect in the plan.
    """
    if plan.empty:
        contradicted = int(classified.verdict.eq('factor_contradicted').sum())
        if contradicted:
            raise D12Error(f'{contradicted} confirmed break(s) but an empty repair plan')
        return {'checked_instruments': 0, 'checked_breaks': 0, 'maximum_corrected_ratio_error': 0.0,
                'maximum_within_segment_shift': 0.0, 'maximum_residual_level_mismatch': 0.0}
    worst_ratio = worst_within = worst_level = 0.0
    checked = 0
    for instrument, group in classified.loc[classified.instrument.isin(plan.instrument)].groupby('instrument'):
        group = group.sort_values('trade_date')
        multiplier = segment_multipliers(plan, instrument, group.trade_date).to_numpy()
        scaled = group.factor.to_numpy() * multiplier
        corrected_ratio = scaled / np.concatenate([[np.nan], scaled[:-1]])
        corrected_scale = group.scale.to_numpy() * multiplier
        observed = group.observed_factor_ratio.to_numpy()
        independent = group.independent_factor_ratio.to_numpy()
        breaks = np.flatnonzero(group.verdict.eq('factor_contradicted').to_numpy())
        for order, position in enumerate(breaks):
            error = abs(corrected_ratio[position] / independent[position] - 1)
            if np.isfinite(error):
                worst_ratio = max(worst_ratio, float(error))
                checked += 1
            low = breaks[order - 1] if order else 0
            high = breaks[order + 1] if order + 1 < len(breaks) else len(corrected_scale)
            before = corrected_scale[max(low, position - window):position]
            after = corrected_scale[position:min(high, position + window)]
            before, after = before[np.isfinite(before)], after[np.isfinite(after)]
            if before.size and after.size:
                worst_level = max(worst_level, abs(float(np.median(after) / np.median(before)) - 1))
        inside = np.ones(len(group), dtype=bool)
        inside[breaks] = False
        inside[0] = False
        if inside.any():
            shift = np.abs(corrected_ratio[inside] / observed[inside] - 1)
            if np.isfinite(shift).any():
                worst_within = max(worst_within, float(np.nanmax(shift)))
    if worst_ratio > ratio_tol:
        raise D12Error(f'Repaired ratio still disagrees with the evidence at a confirmed break: {worst_ratio:.3e}')
    if worst_within > ratio_tol:
        raise D12Error(f'Repair moved a within-segment ratio it must leave alone: {worst_within:.3e}')
    return {'checked_instruments': int(plan.instrument.nunique()), 'checked_breaks': checked,
            'maximum_corrected_ratio_error': worst_ratio,
            'maximum_within_segment_shift': worst_within,
            'maximum_residual_level_mismatch': worst_level,
            'residual_level_note': f'Reported, not gated. A residual up to about {level_tol:.1%} is the '
                                   "provider's own factor wobble, not a defect in the plan."}


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

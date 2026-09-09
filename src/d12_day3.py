"""Day-3 evidence functions. No repairs, fitting, universe changes or R1 orders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.d12_repo import D12Error
from src.d10_core import unique_frame

KEYS = ['trade_date', 'instrument']
COSTS = ['commission_cost', 'transfer_fee_cost', 'stamp_duty_cost', 'impact_cost']


def price_audit(prices, limits, calendar, *, jump_threshold=.30, factor_threshold=.05):
    """Two independent screens, never a board-based assertion of impossible returns.

    Missing/zero limits do NOT establish an unlimited session. A large adjusted
    return is a review candidate, including legitimate resumption/corporate actions.
    Reindex to exchange sessions so a missing row cannot compress the return span.
    """
    unique_frame(prices, KEYS, 'price audit')
    if limits.duplicated(KEYS).any(): raise D12Error('Duplicate price limits')
    dates = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    names = sorted(prices.instrument.unique())
    grid = pd.MultiIndex.from_product([names, dates], names=['instrument', 'trade_date'])
    f = prices.set_index(['instrument', 'trade_date']).reindex(grid).reset_index()
    f = f.merge(limits[KEYS + ['up_limit', 'down_limit']], on=KEYS, how='left', validate='one_to_one')
    for col in ['close', 'factor', 'high', 'low', 'up_limit', 'down_limit']:
        f[col] = pd.to_numeric(f[col], errors='coerce').astype(float)
    good_factor = np.isfinite(f.factor) & f.factor.gt(0)
    for col in ['close', 'high', 'low']:
        f['raw_' + col] = (f[col] / f.factor.where(good_factor)).where(lambda x: np.isfinite(x) & x.gt(0))
    valid = np.isfinite(f.close) & f.close.gt(0)
    f['observed_date'] = f.trade_date.where(valid)
    f['observed_close'] = f.close.where(valid)
    g = f.groupby('instrument', sort=False)
    f['mark'] = g.observed_close.ffill()
    f['mark_source_date'] = g.observed_date.ffill()
    f['previous_mark'] = f.groupby('instrument').mark.shift()
    f['previous_mark_source_date'] = f.groupby('instrument').mark_source_date.shift()
    f['mark_return'] = f.mark / f.previous_mark - 1
    positions = pd.Series(np.arange(len(dates)), index=dates)
    f['return_span_sessions'] = f.mark_source_date.map(positions) - f.previous_mark_source_date.map(positions)
    f['stale_mark_sessions'] = f.trade_date.map(positions) - f.mark_source_date.map(positions)
    f['factor_ratio'] = f.factor.where(good_factor) / f.factor.where(good_factor).groupby(f.instrument).shift()
    f['raw_return_1d'] = f.raw_close / f.groupby('instrument').raw_close.shift() - 1
    bounds = np.isfinite(f.up_limit) & np.isfinite(f.down_limit) & f.down_limit.gt(0) & f.up_limit.ge(f.down_limit)
    f['limit_status'] = np.select(
        [f.up_limit.isna() | f.down_limit.isna(), f.up_limit.eq(0) & f.down_limit.eq(0), bounds],
        ['missing', 'zero_placeholder', 'valid'], default='invalid')
    # One-fen slack covers tick rounding and provider float32; no change to replay tolerances.
    f['outside_raw_limits'] = bounds & ((f.raw_high > f.up_limit + .011) | (f.raw_low < f.down_limit - .011)
                                                        | (f.raw_close > f.up_limit + .011) | (f.raw_close < f.down_limit - .011))
    f['adjusted_jump_candidate'] = f.mark_return.abs().gt(jump_threshold)
    f['factor_jump_candidate'] = (f.factor_ratio - 1).abs().gt(factor_threshold)
    f['resumption_or_gap'] = valid & f.return_span_sessions.gt(1)
    f['invalid_factor_on_price'] = valid & ~good_factor
    f['review_candidate'] = f[['outside_raw_limits', 'adjusted_jump_candidate', 'factor_jump_candidate',
                                'resumption_or_gap', 'invalid_factor_on_price']].any(axis=1)
    f['observed_price'] = valid
    return f


def reference_crosscheck(audit, reference):
    """Only compare factor changes, never provider-specific factor levels."""
    if reference.duplicated(KEYS).any(): raise D12Error('Duplicate independent price reference')
    f = audit.merge(reference, on=KEYS, how='left', validate='one_to_one').sort_values(['instrument', 'trade_date'])
    f['raw_reference_error'] = (f.raw_close / f.reference_close.where(f.reference_close.gt(0)) - 1).abs()
    ref = f.reference_factor.where(f.reference_factor.gt(0))
    f['reference_factor_ratio'] = ref / ref.groupby(f.instrument).shift()
    f['factor_reference_error'] = (f.factor_ratio / f.reference_factor_ratio - 1).abs()
    f['reference_mismatch'] = f.raw_reference_error.gt(1e-4) | f.factor_reference_error.gt(1e-3)
    f['review_candidate'] |= f.reference_mismatch
    return f


def price_propagation(candidates, weights, path, predictions, labels, orders, exposures):
    """Observed direct channels, plus explicitly unexcluded model dependencies.

    No direct holdings is not evidence of B-only. This is exposure tracing, not
    a counterfactual estimate of the return change from correcting the prices.
    """
    rows = []
    wb = weights.set_index(KEYS).weight
    by_pred = {n:g for n,g in predictions.groupby('instrument')}
    by_label = {n:g for n,g in labels.groupby('instrument')}
    by_order = {n:g for n,g in orders.groupby('instrument')}
    by_exposure = {n:g for n,g in exposures.groupby('instrument')}
    for e in candidates.itertuples(index=False):
        d, n = pd.Timestamp(e.trade_date), e.instrument
        beginning = float(path.beginning_weights.at[d, n]) if d in path.beginning_weights.index and n in path.beginning_weights else np.nan
        ending = float(path.ending_weights.at[d, n]) if d in path.ending_weights.index and n in path.ending_weights else np.nan
        p = by_pred.get(n, predictions.iloc[:0])
        lab = by_label.get(n, labels.iloc[:0])
        crossing = lab.loc[lab.label_start_timestamp.le(d) & lab.label_end_timestamp.ge(d)]
        order = by_order.get(n, orders.iloc[:0])
        impacted_orders = order.loc[order.signal_date.eq(d) | order.execution_date.eq(d)]
        ex = by_exposure.get(n, exposures.iloc[:0])
        ex = ex.loc[ex.trade_date.ge(d)]
        rows.append({'trade_date': d, 'instrument': n, 'mark_return': e.mark_return,
            'beginning_actual_weight': beginning, 'ending_actual_weight': ending,
            'holding_pnl_nav_fraction_observed': beginning * e.mark_return,
            'benchmark_weight_same_day': wb.get((d, n), np.nan),
            'orders_on_signal_or_execution': len(impacted_orders),
            'predictions_on_or_after': int(p.sample_date.ge(d).sum()),
            'label_windows_crossing_date': len(crossing),
            'later_exposure_rows': len(ex),
            'classification': 'A_possible_pending_evidence',
            'unexcluded_channels': 'feature lookbacks; cross-sectional transforms; matured training labels; calibration; normalized benchmark drift; risk history',
            'interpretation': 'observed exposure only; not error-attributable PnL; no-holding does not imply B'})
    return pd.DataFrame(rows)


def membership_timing(weights, intervals, snapshots, calendar):
    """Compare provider intervals, project weights and observed snapshots, without inferring official dates."""
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    members = weights.groupby('trade_date').instrument.agg(set)
    snapshot_dates = set(pd.to_datetime(snapshots.trade_date))
    rows, old, provider_old = [], None, None
    for d, current in members.items():
        provider = set(intervals.loc[intervals.start_date.le(d) & intervals.end_date.ge(d), 'instrument'])
        rows.append({'trade_date': d, 'processed_members': len(current), 'provider_members': len(provider),
            'processed_added': len(current - old) if old is not None else 0,
            'processed_removed': len(old - current) if old is not None else 0,
            'provider_added': len(provider - provider_old) if provider_old is not None else 0,
            'provider_removed': len(provider_old - provider) if provider_old is not None else 0,
            'provider_only': '|'.join(sorted(provider-current)), 'processed_only': '|'.join(sorted(current-provider)),
            'snapshot_on_date': d in snapshot_dates,
            'session_in_month': int(((dates.to_period('M') == d.to_period('M')) & (dates <= d)).sum()),
            'official_timing_confirmed': False})
        old, provider_old = current, provider
    return pd.DataFrame(rows)


def suspected_membership_windows(weights, path, calendar):
    """Next observed basket is a retrospective proxy, NOT an official constituent list.

    Candidate dates follow the day-2 hypothesis only. No proxy enters any order,
    return, signal, repaired weight, or official weight metric.
    """
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    sets = weights.groupby('trade_date').instrument.agg(set).sort_index()
    rows = []
    for year in sorted(set(dates.year)):
        for month in [6, 12]:
            fridays = pd.date_range(f'{year}-{month:02d}-01', periods=28, freq='D')
            friday = fridays[fridays.dayofweek == 4][1]
            after = dates[dates > friday]
            if len(after) == 0: continue
            candidate = after[0]
            before = sets.loc[sets.index < candidate]
            if before.empty: continue
            old = before.iloc[-1]
            later = sets.loc[(sets.index >= candidate) & (sets.index <= candidate + pd.Timedelta(days=45))]
            changed = [(d, s) for d, s in later.items() if s != old]
            if not changed: continue
            caught_up, new = changed[0]
            proxy_out, proxy_in = old - new, new - old
            for d in path.ending_weights.index[(path.ending_weights.index >= candidate) & (path.ending_weights.index < caught_up)]:
                current = sets.loc[d]
                rows.append({'trade_date': d, 'candidate_effective_date': candidate, 'next_basket_change': caught_up,
                    'proxy_removed_actual_weight': float(path.ending_weights.loc[d].reindex(sorted(proxy_out)).fillna(0).sum()),
                    'proxy_added_not_in_universe': len(proxy_in - current),
                    'proxy_removed_names': '|'.join(sorted(proxy_out)), 'proxy_added_names': '|'.join(sorted(proxy_in)),
                    'official_added_weight': np.nan, 'official_common_weight_difference': np.nan,
                    'classification': 'retrospective_next_basket_proxy_not_official'})
    return pd.DataFrame(rows)


def official_membership_comparison(evidence, weights, path):
    """Optional independently evidenced FULL sets, not a list of adds/removes.

    available_at is a timestamp; a post-close announcement cannot be used for that
    day's close decision. We conservatively use it from the next calendar date.
    Official membership and historical information availability remain separate.
    """
    required = ['effective_date', 'available_at', 'instrument', 'source_url']
    if not set(required).issubset(evidence): raise D12Error('membership evidence needs '+str(required))
    e = evidence.copy()
    e.effective_date = pd.to_datetime(e.effective_date).dt.normalize()
    e.available_at = pd.to_datetime(e.available_at, utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    unique_frame(e, ['effective_date', 'instrument'], 'official full membership sets')
    if e.source_url.isna().any() or e.source_url.str.strip().eq('').any(): raise D12Error('Missing membership evidence URL')
    sets = {d: g for d, g in e.groupby('effective_date')}
    rows = []
    for d, w in weights.groupby('trade_date'):
        prior = [day for day in sets if day <= d]
        if not prior or d not in path.ending_weights.index: continue
        frame = sets[max(prior)]
        if len(frame) != len(w): raise D12Error('Evidence must contain the full constituent set, matching registered index size')
        official, actual = set(frame.instrument), set(w.instrument)
        outgoing = actual - official
        held_outside = path.ending_weights.loc[d, ~path.ending_weights.columns.isin(official)].sum()
        rows.append({'trade_date': d, 'effective_date': frame.effective_date.iloc[0],
            'latest_available_at': frame.available_at.max(),
            'available_before_decision_day': bool(frame.available_at.max().normalize() < d),
            'removed_still_in_universe': len(outgoing), 'added_missing_from_universe': len(official-actual),
            'actual_holdings_outside_official_weight': float(held_outside),
            'official_added_weight': np.nan,
            'note': 'full-list evidence only; official weights unavailable; path-wide effect remains unquantified'})
    return pd.DataFrame(rows)


def trade_groups(trades, targets, predictions, audits, estimates, market, *, participation_limit=.05):
    """P1 side/type/strength/return groups plus causal participation telemetry."""
    keys = ['signal_date', 'instrument']
    unique_frame(trades, keys, 'P1 fills')
    unique_frame(targets, keys, 'P1 targets')
    f = trades.merge(targets[keys+['expected_excess']].assign(current_member=True),
                     on=keys, how='left', validate='one_to_one')
    p = predictions.rename(columns={'sample_date': 'signal_date'})
    f = f.merge(p[keys+['score_z']], on=keys, how='left', validate='one_to_one')
    f = f.merge(audits[['signal_date', 'initial_construction_exempt']], on='signal_date', how='left', validate='many_to_one')
    f['side'] = np.where(f.order_units.gt(0), 'buy', 'sell')
    f['trade_type'] = np.select([f.initial_construction_exempt.eq(True), ~f.current_member.eq(True)],
                                ['initial', 'forced_exit'], default='active')
    if f.loc[f.trade_type.eq('forced_exit'), 'order_units'].gt(0).any(): raise D12Error('Nonmember buy is not a forced exit')
    f['score_strength'] = pd.cut(f.score_z.abs(), [-np.inf,.5,1.,2.,np.inf], labels=['<0.5','0.5-1','1-2','>=2'], right=False).astype(object).fillna('unavailable')
    f['expected_return_group'] = pd.cut(f.expected_excess, [-np.inf,-.005,0,.005,np.inf],
                                       labels=['<-50bp','-50bp..0','0..50bp','>=50bp'], right=False).astype(object).fillna('unavailable')
    f['filled'] = f.executed_units.ne(0)
    f['requested_signal_cny'] = f.order_units.abs()*f.signal_price
    f['filled_cny'] = f.executed_trade_amount.abs()
    f['unfilled_signal_cny'] = f.unfilled_units.abs()*f.signal_price
    f['expected_trade_contribution'] = f.order_units*f.signal_price/f.signal_nav*f.expected_excess
    e = estimates.rename(columns={'trade_date':'signal_date'})
    f = f.merge(e[keys+['a_est','sigma_est','sigma_source']], on=keys, how='left', validate='one_to_one')
    valid = np.isfinite(f.a_est) & f.a_est.gt(0)
    f['ex_ante_participation'] = f.requested_signal_cny / f.a_est.where(valid)
    f['ex_ante_above_limit'] = f.ex_ante_participation.gt(participation_limit)
    f['a_est_missing'] = ~valid
    # Keep realized market state on a separate telemetry merge, never used in A_est.
    q = market.rename(columns={'trade_date':'execution_date'})
    f = f.merge(q[['execution_date','instrument','daily_amount_cny']], on=['execution_date','instrument'], how='left', validate='one_to_one')
    f['realized_participation'] = f.filled_cny/f.daily_amount_cny.where(f.daily_amount_cny.gt(0))
    capacity = f.daily_amount_cny*participation_limit/f.execution_price.where(f.execution_price.gt(0))
    eligible = np.where(f.side.eq('buy'), f.exchange_can_buy.eq(True), f.exchange_can_sell.eq(True))
    f['capacity_shortfall_cny'] = ((f.order_units.abs()-capacity).clip(lower=0)*f.execution_price).where(eligible,0.)
    f['cash_scaled'] = f.side.eq('buy') & f.cash_scale.lt(1.-1e-12) & f.filled
    # Actual forced turnover: conventionally full forced sell/NAV (not half).
    f['total_turnover_component'] = .5*f.filled_cny/f.signal_nav
    f['mandatory_turnover_component'] = (f.filled_cny/f.signal_nav).where(f.trade_type.eq('forced_exit'), 0.)
    aggregations = {'order_rows':('instrument','size'), 'filled_rows':('filled','sum'),
        'requested_signal_cny':('requested_signal_cny','sum'), 'filled_cny':('filled_cny','sum'),
        'unfilled_signal_cny':('unfilled_signal_cny','sum'), 'cash_scaled_rows':('cash_scaled','sum'),
        'ex_ante_above_limit_rows':('ex_ante_above_limit','sum'), 'a_est_missing_rows':('a_est_missing','sum'),
        'max_ex_ante_participation':('ex_ante_participation','max'),
        'capacity_limited_rows':('participation_limited','sum'), 'capacity_shortfall_cny':('capacity_shortfall_cny','sum'),
        'expected_contribution_known_rows':('expected_trade_contribution','count'),
        'expected_trade_contribution_sum':('expected_trade_contribution',lambda x:x.sum(min_count=1)),
        'total_cost':('total_cost','sum'), **{c:(c,'sum') for c in COSTS}}
    groups = f.groupby(['side','trade_type','score_strength','expected_return_group'], observed=True, dropna=False).agg(**aggregations).reset_index()
    by_date = f.groupby(['signal_date','side'], observed=True).agg(**aggregations).reset_index()
    turnover = f.groupby('signal_date').agg(actual_total_turnover=('total_turnover_component','sum'),
                                          actual_mandatory_turnover=('mandatory_turnover_component','sum'))
    turnover['actual_active_turnover'] = (turnover.actual_total_turnover-turnover.actual_mandatory_turnover).clip(lower=0)
    decisions = audits.merge(turnover.reset_index(), on='signal_date', how='left', validate='one_to_one')
    for c in COSTS+['total_cost']:
        if not np.isclose(groups[c].sum(), trades[c].sum(), rtol=1e-12, atol=1e-6): raise D12Error('P1 group costs fail reconciliation: '+c)
    return f, groups, by_date, decisions

"""Diagnostics of saved D11 results; no fitting, parameter selection or new OOS."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.d3_core import trade_cost
from src.d9_core import aggregate_daily_to_monthly, maximum_relative_drawdown, summarize_monthly_performance
from src.d10_core import (DiagnosticError, ExecutedPath, linked_monthly, require_close,
                         unique_frame, wls_cross_section)

COST_COLUMNS = ['commission_cost', 'transfer_fee_cost', 'stamp_duty_cost', 'impact_cost']


def valid_daily(frame):
    frame = frame.copy().sort_values('trade_date').reset_index(drop=True)
    unique_frame(frame, ['trade_date'], 'daily returns')
    if frame.empty or not np.isfinite(frame[['portfolio_return', 'benchmark_return']]).all().all():
        raise DiagnosticError('Empty or nonfinite daily returns')
    if (frame[['portfolio_return', 'benchmark_return']] <= -1).any().any():
        raise DiagnosticError('Daily return <= -100%')
    return frame


def d11_price_panels(market, start):
    """D11 marks use positive closes only, with forward fill starting at OOS.

    Warmup closes are retained for first-day benchmark attribution, but must
    never seed missing OOS marks: simulate_causal starts its mark state empty.
    Execution quotes are raw positive VWAP, with NO close fallback.
    """
    unique_frame(market, ['trade_date', 'instrument'], 'market')
    close = market.pivot(index='trade_date', columns='instrument', values='close').sort_index().astype(float)
    close = close.where(close.gt(0) & np.isfinite(close))
    marks = close.ffill()
    marks.loc[marks.index >= start] = close.loc[close.index >= start].ffill()
    vwap = market.pivot(index='trade_date', columns='instrument', values='vwap').reindex_like(close).astype(float)
    return marks, vwap.where(vwap.gt(0) & np.isfinite(vwap))


def reconstruct_d11(daily, trades, market, initial_capital, costs):
    """Independently rebuild inventory/cash from signed executed units.

    Verify signal inventory, quantities, per-order costs, daily cash/NAV,
    realized turnover and holding + execution - costs PnL identity.
    """
    daily = valid_daily(daily)
    unique_frame(trades, ['execution_date', 'instrument'], 'D11 trades')
    if not set(trades.execution_date).issubset(set(daily.trade_date)):
        raise DiagnosticError('Trades outside daily calendar')
    marks, prices = d11_price_panels(market, daily.trade_date.min())
    if not pd.Index(daily.trade_date).isin(marks.index).all():
        raise DiagnosticError('Missing market session')
    names = marks.columns.union(pd.Index(trades.instrument.unique()))
    marks, prices = marks.reindex(columns=names), prices.reindex(columns=names)
    units = pd.Series(0., index=names)
    cash = previous_nav = float(initial_capital)
    groups = {d: g.set_index('instrument') for d, g in trades.groupby('execution_date')}
    quotes = {d: g.set_index('instrument') for d, g in market.groupby('trade_date')}
    beginning, changes, ending, rows, order_rows = [], [], [], [], []
    last_marks = pd.Series(np.nan, index=names)
    for row in daily.itertuples(index=False):
        date = row.trade_date
        mp, ep = marks.loc[date], prices.loc[date]
        held = units > 0
        if mp[held].isna().any() or last_marks[held].isna().any():
            raise DiagnosticError(f'{date}: unpriceable held units')
        bw = units * last_marks.fillna(0) / previous_nav
        holding = float((units * (mp-last_marks).fillna(0)).sum()) / previous_nav
        tw = pd.Series(0., index=names)
        fee_parts = dict.fromkeys(COST_COLUMNS, 0.)
        turnover = 0.
        g = groups.get(date)
        if g is not None:
            if g.signal_date.nunique() != 1 or g.signal_nav.nunique() != 1:
                raise DiagnosticError('Mixed signal or NAV in execution group')
            require_close(g.signal_units, units.reindex(g.index), 'signal inventory', atol=1e-7)
            require_close(g.signal_nav, np.full(len(g), previous_nav), 'signal NAV', atol=1e-4)
            require_close(g.signal_price, last_marks.reindex(g.index), 'signal prices')
            signed = g.executed_units.astype(float)
            wanted = g.order_units.astype(float)
            if ((signed * wanted < 0) | (signed.abs() > wanted.abs()+1e-8)).any():
                raise DiagnosticError('Fill exceeds or reverses fixed order')
            filled = signed.ne(0)
            if ep.reindex(g.index[filled]).isna().any():
                raise DiagnosticError('Filled trade without valid VWAP')
            require_close(g.loc[filled, 'execution_price'], ep.reindex(g.index[filled]), 'execution quotes')
            amount = signed * ep.reindex(g.index).fillna(0)
            require_close(g.executed_trade_amount, amount, 'units x VWAP', atol=1e-5)
            require_close(g.unfilled_units, wanted-signed, 'unfilled units', atol=1e-7)
            for n, order in g.iterrows():
                quote = quotes[date].reindex([n]).iloc[0]
                if ((order.executed_units > 0 and quote.exchange_can_buy != True)
                        or (order.executed_units < 0 and quote.exchange_can_sell != True)):
                    raise DiagnosticError('Exchange-prohibited execution')
                recomputed = trade_cost(order.executed_trade_amount, quote.volatility,
                                        quote.daily_amount_cny, date, costs)
                for key, value in recomputed.items():
                    require_close([order[key]], [value], f'{date}/{n}/{key}', atol=1e-5)
                for key in COST_COLUMNS:
                    fee_parts[key] += recomputed[key]
                post_close = float(order.executed_units * (mp.get(n)-ep.get(n))) if order.executed_units else 0.
                # A marked opportunity estimate, not a realizable counterfactual.
                unfilled_close = (float(order.unfilled_units*(mp.get(n)-ep.get(n)))
                                  if pd.notna(mp.get(n)) and pd.notna(ep.get(n)) else np.nan)
                order_rows.append({'execution_date': date, 'signal_date': order.signal_date,
                    'instrument': n, 'post_execution_pnl_cny': post_close,
                    'unfilled_to_close_mark_cny': unfilled_close})
            old = units.reindex(g.index).copy()
            units.loc[g.index] += signed
            dust = signed.lt(0) & units.reindex(g.index).abs().le(np.maximum(old.abs()*1e-12, 1e-12))
            units.loc[g.index[dust]] = 0.
            if (units < 0).any():
                raise DiagnosticError('Negative reconstructed inventory')
            cost = sum(fee_parts.values())
            cash -= float(amount.sum()) + cost
            if cash < -max(1e-7, previous_nav*2e-9):
                raise DiagnosticError('Negative reconstructed cash')
            cash = max(cash, 0.)
            tw.loc[g.index] = amount / previous_nav
            turnover = .5*float(amount.abs().sum()) / float(g.signal_nav.iloc[0])
        cost = sum(fee_parts.values())
        if mp[units.gt(0)].isna().any():
            raise DiagnosticError('Unpriceable ending inventory')
        nav = float(cash+(units*mp.fillna(0)).sum())
        execution = float((tw*(mp/prices.loc[date]-1).fillna(0)).sum())
        error = require_close([nav, cash, cost], [row.nav, row.cash, row.transaction_cost], 'NAV/cash/cost', atol=1e-4)
        require_close([turnover], [row.single_sided_turnover], 'actual turnover')
        require_close([row.portfolio_return], [row.nav/previous_nav-1], 'saved return')
        residual = row.portfolio_return-holding-execution+cost/previous_nav
        require_close([residual], [0.], 'PnL closure')
        beginning.append(bw.to_numpy()); changes.append(tw.to_numpy())
        ending.append((units*mp.fillna(0)/nav).to_numpy())
        rows.append({'trade_date': date, 'previous_nav': previous_nav,
            'holding_return': holding, 'execution_return': execution,
            'cost_return': cost/previous_nav, 'accounting_residual': residual,
            'nav_error_cny': error, 'cash_weight': cash/nav, **fee_parts})
        previous_nav, last_marks = float(row.nav), mp
    dates = pd.DatetimeIndex(daily.trade_date, name='trade_date')
    panel = lambda v: pd.DataFrame(v, index=dates, columns=names)
    return ExecutedPath(pd.DataFrame(rows), panel(beginning), panel(changes), panel(ending), {}, pd.DataFrame(order_rows))


def performance_and_costs(daily, initial_capital, lags):
    daily = valid_daily(daily)
    net, monthly = summarize_monthly_performance(daily, newey_west_lags=lags)
    gross_daily = daily.copy()
    gross_daily.portfolio_return += daily.transaction_cost / daily.nav.shift(1, fill_value=initial_capital)
    gross, gross_monthly = summarize_monthly_performance(gross_daily, newey_west_lags=lags)
    monthly['same_path_gross_excess'] = gross_monthly.excess_return
    monthly['cost_drag'] = monthly.same_path_gross_excess-monthly.excess_return
    years = []
    for year, g in daily.groupby(daily.trade_date.dt.year):
        m = monthly.loc[monthly.month.str.startswith(str(year))]
        p, b = float((1+g.portfolio_return).prod()-1), float((1+g.benchmark_return).prod()-1)
        te = float(m.excess_return.std(ddof=1)*np.sqrt(12)) if len(m)>1 else None
        ann = float(m.excess_return.mean()*12)
        years.append({'year': int(year), 'start': str(g.trade_date.min().date()), 'end': str(g.trade_date.max().date()),
            'months': len(m), 'partial_year': len(m)!=12, 'portfolio_return': p, 'benchmark_return': b,
            'period_excess_difference': p-b, 'annualized_monthly_excess': ann,
            'information_ratio': ann/te if te and te>0 else None,
            'maximum_relative_drawdown': maximum_relative_drawdown(g),
            'positive_month_fraction': float(m.excess_return.gt(0).mean()),
            'cost_drag_annualized': float(m.cost_drag.mean()*12),
            'transaction_cost_cny': float(g.transaction_cost.sum()),
            'sum_executed_single_sided_turnover': float(g.single_sided_turnover.sum()),
            'mean_cash_weight': float((g.cash/g.nav).mean())})
    streak = longest = 0
    for negative in monthly.excess_return.lt(0):
        streak = streak+1 if negative else 0
        longest = max(longest, streak)
    relative = np.r_[1., ((1+daily.portfolio_return).cumprod()/(1+daily.benchmark_return).cumprod()).to_numpy()]
    trough = int(np.argmin(relative/np.maximum.accumulate(relative)-1))
    peak = int(np.argmax(relative[:trough+1]))
    recoveries = np.flatnonzero(relative[trough+1:] >= relative[peak])
    date_at = lambda i: 'initial_capital' if i==0 else str(daily.trade_date.iloc[i-1].date())
    summary = {'net': net, 'same_path_gross': gross,
        'cost_drag_annualized': gross['annualized_excess_return']-net['annualized_excess_return'],
        'gross_definition': 'daily fees added back on realized path; not a zero-cost rerun',
        'longest_negative_month_streak': longest,
        'drawdown_peak': date_at(peak), 'drawdown_trough': date_at(trough),
        'drawdown_recovery': date_at(trough+1+int(recoveries[0])) if len(recoveries) else None,
        'top_five_months_excess_sum': float(monthly.excess_return.nlargest(5).sum()),
        'other_months_mean_excess': float(monthly.excess_return.sort_values().iloc[:-5].mean()) if len(monthly)>5 else None}
    return summary, monthly, pd.DataFrame(years)


def signal_diagnostics(predictions, labels, end):
    keys = ['sample_date', 'instrument']
    unique_frame(predictions, keys, 'predictions'); unique_frame(labels, keys, 'labels')
    # LEFT join exclusively for evaluation; this function never emits new signals.
    f = predictions.merge(labels[keys+['excess_return', 'label_end_timestamp']], on=keys,
                          how='left', validate='one_to_one')
    f['evaluable'] = (f.label_end_timestamp.le(pd.Timestamp(end)) & np.isfinite(f.excess_return)
                       & np.isfinite(f.prediction) & f.feature_available.eq(True))
    rows = []
    for date, g in f.groupby('sample_date'):
        x = g.loc[g.evaluable]
        eligible = len(x)>=3 and x.prediction.nunique()>1 and x.excess_return.nunique()>1
        ic = float(x.prediction.corr(x.excess_return, method='spearman')) if eligible else None
        spread = None
        if eligible and len(x)>=10:
            # Average-rank percentile avoids selecting tied names by ticker order.
            rank = x.prediction.rank(method='average', pct=True)
            top, bottom = x.loc[rank.gt(.9), 'excess_return'], x.loc[rank.le(.1), 'excess_return']
            if len(top) and len(bottom): spread = float(top.mean()-bottom.mean())
        rows.append({'sample_date': date, 'predicted_rows': len(g), 'evaluated_rows': len(x),
            'missing_or_immature_label_rows': int((g.label_end_timestamp.gt(pd.Timestamp(end)) |
                g.label_end_timestamp.isna() | ~np.isfinite(g.excess_return)).sum()),
            'feature_abstention_rows': int((~g.feature_available.eq(True)).sum()),
            'rank_ic': ic, 'top_bottom_label_spread': spread})
    dates = pd.DataFrame(rows)
    years = []
    for year, g in dates.groupby(dates.sample_date.dt.year):
        ic = pd.to_numeric(g.rank_ic).dropna()
        sd = float(ic.std(ddof=1)) if len(ic)>1 else None
        years.append({'year': int(year), 'prediction_dates': len(g), 'ic_dates': len(ic),
            'rank_ic_mean': float(ic.mean()) if len(ic) else None,
            'icir_unannualized': float(ic.mean()/sd) if sd and sd>0 else None,
            'rank_ic_positive_fraction': float(ic.gt(0).mean()) if len(ic) else None,
            'evaluation_coverage': float(g.evaluated_rows.sum()/g.predicted_rows.sum()),
            'mean_top_bottom_label_spread': float(pd.to_numeric(g.top_bottom_label_spread).mean())})
    return dates, pd.DataFrame(years)


def execution_diagnostics(trades, targets, audits, path):
    f = trades.copy()
    unique_frame(f, ['signal_date','instrument'], 'trades')
    requested = f.order_units.abs()*f.signal_price
    filled = f.executed_units.abs()*f.signal_price
    f['requested_at_signal_cny'], f['filled_at_signal_cny'] = requested, filled
    f['unfilled_at_signal_cny'] = requested-filled
    f['exchange_blocked'] = ((f.order_units.gt(0)&~f.exchange_can_buy.eq(True)) |
                              (f.order_units.lt(0)&~f.exchange_can_sell.eq(True)))
    f['cash_scaled_buy'] = f.order_units.gt(0)&f.executed_units.gt(0)&f.cash_scale.lt(1-1e-12)
    membership = targets[['signal_date','instrument']].assign(current_member=True)
    f = f.merge(membership, on=['signal_date','instrument'], how='left', validate='one_to_one')
    f['forced_exit'] = f.current_member.isna()&f.order_units.lt(0)
    f = f.merge(path.orders, on=['execution_date','signal_date','instrument'], how='left', validate='one_to_one')
    rows = []
    for signal, g in f.groupby('signal_date'):
        rows.append({'signal_date': signal, 'execution_date': g.execution_date.iloc[0],
            'order_records': len(g), 'filled_records': int(g.executed_units.ne(0).sum()),
            'requested_at_signal_cny': float(g.requested_at_signal_cny.sum()),
            'fill_fraction_at_signal_prices': float(g.filled_at_signal_cny.sum()/g.requested_at_signal_cny.sum()),
            'unfilled_at_signal_cny': float(g.unfilled_at_signal_cny.sum()),
            'exchange_blocked_records': int(g.exchange_blocked.sum()),
            'participation_limited_records': int(g.participation_limited.sum()),
            'cash_scaled_buy_records': int(g.cash_scaled_buy.sum()),
            'executed_total_turnover': .5*float(g.executed_trade_amount.abs().sum())/float(g.signal_nav.iloc[0]),
            'forced_exit_sold_nav_fraction': float(-g.loc[g.forced_exit,'executed_trade_amount'].sum()/g.signal_nav.iloc[0]),
            'forced_exit_cost_cny': float(g.loc[g.forced_exit,'total_cost'].sum()),
            'unfilled_to_close_mark_cny': float(g.unfilled_to_close_mark_cny.sum()),
            'unfilled_mark_unavailable_records': int(g.unfilled_to_close_mark_cny.isna().sum()),
            **{k: float(g[k].sum()) for k in COST_COLUMNS+['total_cost']}})
    detail = pd.DataFrame(rows).merge(audits[['signal_date','initial_construction_exempt','target_active_turnover',
        'target_total_turnover']], on='signal_date', how='left', validate='one_to_one')
    if detail.initial_construction_exempt.isna().any(): raise DiagnosticError('Execution missing target audit')
    regular = detail.loc[~detail.initial_construction_exempt.astype(bool)]
    summary = {'order_records': len(f), 'filled_records': int(f.executed_units.ne(0).sum()),
        'fill_fraction_at_signal_prices': float(filled.sum()/requested.sum()),
        'regular_mean_target_active_turnover': float(regular.target_active_turnover.mean()),
        'regular_mean_executed_total_turnover': float(regular.executed_total_turnover.mean()),
        'forced_exit_cost_cny': float(detail.forced_exit_cost_cny.sum()),
        'total_cost_cny': float(f.total_cost.sum()), 'costs_cny': {k:float(f[k].sum()) for k in COST_COLUMNS},
        'reason_counts_overlap': True,
        'unfilled_mark_definition': 'signed unfilled units x (execution-day close - VWAP); descriptive, excludes costs and feasibility; not a causal loss'}
    return summary, detail


def compare_common_dates(old, new, official, lags):
    old, new = valid_daily(old), valid_daily(new)
    unique_frame(official, ['trade_date'], 'official returns')
    joined = old[['trade_date','portfolio_return','benchmark_return']].merge(
        new[['trade_date','portfolio_return','benchmark_return']], on='trade_date', suffixes=('_old','_new'), validate='one_to_one')
    if joined.empty: raise DiagnosticError('No common comparison dates')
    # Intersection must not hide internal gaps; dropped leading/trailing days are disclosed.
    lo, hi = joined.trade_date.min(), joined.trade_date.max()
    expected = official.loc[official.trade_date.between(lo,hi), 'trade_date']
    if not pd.DatetimeIndex(joined.trade_date).equals(pd.DatetimeIndex(expected)):
        raise DiagnosticError('Internal calendar gaps in old/new comparison')
    joined = joined.merge(official[['trade_date','benchmark_return']], on='trade_date', validate='one_to_one')
    a = joined[['trade_date','benchmark_return']].assign(portfolio_return=joined.portfolio_return_old)
    b = joined[['trade_date','benchmark_return']].assign(portfolio_return=joined.portfolio_return_new)
    pa, ma = summarize_monthly_performance(a,newey_west_lags=lags)
    pb, mb = summarize_monthly_performance(b,newey_west_lags=lags)
    monthly = ma[['month','trade_days','portfolio_return','excess_return']].merge(
        mb[['month','portfolio_return','excess_return']],on='month',suffixes=('_old','_new'))
    summary = {'common_start': str(lo.date()), 'common_end': str(hi.date()), 'common_days': len(joined),
        'old_excluded_dates': [str(d.date()) for d in old.trade_date[~old.trade_date.isin(joined.trade_date)]],
        'new_excluded_dates': [str(d.date()) for d in new.trade_date[~new.trade_date.isin(joined.trade_date)]],
        'old_saved_benchmark_max_difference': float((joined.benchmark_return_old-joined.benchmark_return).abs().max()),
        'new_saved_benchmark_max_difference': float((joined.benchmark_return_new-joined.benchmark_return).abs().max()),
        'old_common_official_benchmark': pa, 'new_common_official_benchmark': pb,
        'annualized_excess_delta': pb['annualized_excess_return']-pa['annualized_excess_return'],
        'interpretation': 'common daily dates and identical official benchmark; inherited holding histories differ; not single-fix attribution'}
    return summary, monthly


def attribute_d11(daily, path, market, exposures, industries, styles):
    """Lagged daily WLS for close-to-close and supplementary VWAP-to-close.

    The second regression attributes the intraday PnL of actual trades, rather
    than assigning an entire close-to-close return to a newly bought holding.
    Both regressions fit only previous-day constituents. Removed holdings use
    the last exposure known strictly before the return date, with stale
    exposure contribution reported. New names without lagged exposures and
    industries not represented in the fit remain an UNATTRIBUTED bridge.
    """
    unique_frame(exposures, ["trade_date", "instrument"], "exposures")
    exposures = exposures.sort_values(["trade_date", "instrument"])
    mark, execution = d11_price_panels(market, daily.trade_date.min())
    cc = mark / mark.shift(1) - 1
    intraday = mark / execution - 1
    exposure_groups = {pd.Timestamp(d): g.set_index("instrument") for d, g in exposures.groupby("trade_date")}
    all_dates = mark.index
    factor_cols = list(industries) + list(styles)
    latest = pd.DataFrame(columns=factor_cols + ["market_cap", "exposure_date"])
    rows, fits, factors_rows = [], [], []
    accounting = path.daily.set_index("trade_date")
    # Seed only history strictly earlier than the first measured return.
    for d in sorted(d for d in exposure_groups if d < pd.Timestamp(daily.trade_date.min())):
        g = exposure_groups[d][factor_cols + ["market_cap"]].copy()
        g["exposure_date"] = d
        latest = g.copy() if latest.empty else pd.concat([latest.loc[~latest.index.isin(g.index)], g])
    for row in daily.sort_values("trade_date").itertuples(index=False):
        date = pd.Timestamp(row.trade_date)
        pos = all_dates.get_loc(date)
        if pos == 0:
            raise DiagnosticError("attribution needs a previous trading day")
        lag = all_dates[pos - 1]
        if lag not in exposure_groups:
            raise DiagnosticError(f"{date}: missing DAILY lagged exposure at {lag}")
        fit = exposure_groups[lag]
        if "benchmark_weight" not in fit:
            raise DiagnosticError("daily exposure missing benchmark weights")
        wb = fit.benchmark_weight.astype(float)
        require_close([wb.sum()], [1.0], "lagged benchmark sum")
        update = fit[factor_cols + ["market_cap"]].copy(); update["exposure_date"] = lag
        latest = pd.concat([latest.loc[~latest.index.isin(update.index)], update])
        active = path.beginning_weights.loc[date].sub(wb, fill_value=0)
        tw = path.trade_weights.loc[date]
        needed = active.index[active.abs() > 1e-14].union(tw.index[tw.abs() > 1e-14])
        identified = latest.index.intersection(needed)
        unseen = needed.difference(identified)
        ind_active = [c for c in industries if fit[c].abs().sum() > 0]
        # Unsupported industry cannot be estimated from prior constituents.
        supported = latest.loc[identified, ind_active].sum(axis=1).gt(0.5)
        unseen = unseen.union(identified[~supported]); identified = identified[supported]
        contrib = {"industry": 0.0, "style": 0.0, "specific_gross": 0.0}
        stale_l1, stale_pnl, unmodelled, blind_weight = 0.0, 0.0, 0.0, 0.0
        touched = fit.index.union(identified).union(unseen)
        for period, returns, weights in [("close_to_close", cc.loc[date], active),
                                          ("execution_to_close", intraday.loc[date], tw)]:
            if period == "execution_to_close" and not (weights.abs() > 1e-14).any():
                continue
            # A constituent whose return is unobservable this period -- a name the
            # index still carries while Qlib has no bar for it, as when a merger
            # ends its listing -- cannot be regressed or attributed. Dropping it is
            # disclosed, not silent: it leaves the fitted universe, the contribution
            # buckets AND the basket reconstruction together. Those two removals
            # cancel exactly in the closure identity below, so its true index
            # contribution stays inside the official-index bridge, which is never
            # relabelled as stock-specific alpha. A name the PORTFOLIO holds must
            # have a return; that remains fatal.
            missing = returns.reindex(touched)
            blind = missing.index[missing.isna()]
            held_now = path.beginning_weights.loc[date] if period == "close_to_close" else weights
            if (held_now.reindex(blind, fill_value=0).abs() > 1e-14).any():
                raise DiagnosticError(f"{date}: missing realized attribution return for a held name")
            estimable, usable = fit.drop(index=blind, errors="ignore"), identified.difference(blind)
            f, audit = wls_cross_section(estimable, returns, industries, styles)
            fits.append({"trade_date": date, "exposure_date": lag, "period": period, **audit,
                         "unobservable_stocks": int(len(blind)),
                         "unobservable_benchmark_weight": float(wb.reindex(blind, fill_value=0).sum())})
            factors_rows.extend({"trade_date": date, "exposure_date": lag, "period": period,
                                 "factor": name, "factor_return": float(value)} for name, value in f.items())
            w = weights.reindex(usable, fill_value=0)
            x = latest.loc[usable, factor_cols].astype(float)
            r = returns.reindex(usable)
            if r[w.abs() > 1e-14].isna().any():
                raise DiagnosticError(f"{date}: missing realized attribution return")
            r = r.fillna(0)
            industry_r = x[list(industries)] @ f[list(industries)]
            style_r = x[list(styles)] @ f[list(styles)]
            contrib["industry"] += float(w @ industry_r)
            contrib["style"] += float(w @ style_r)
            contrib["specific_gross"] += float(w @ (r - industry_r - style_r))
            stale = pd.to_datetime(latest.loc[usable, "exposure_date"]).lt(lag)
            stale_l1 += float(w[stale].abs().sum()); stale_pnl += float(w[stale] @ r[stale])
            still_unseen = unseen.difference(blind)
            uw = weights.reindex(still_unseen, fill_value=0)
            ur = returns.reindex(still_unseen)
            if ur[uw.abs() > 1e-14].isna().any():
                raise DiagnosticError("unmodelled holding also lacks realized price")
            unmodelled += float(uw @ ur.fillna(0))
            blind_weight += float(weights.reindex(blind, fill_value=0).abs().sum())
        daily_cc = cc.loc[date].reindex(wb.index)
        observable = daily_cc.notna()
        basket_return = float(wb[observable] @ daily_cc[observable])
        unobservable_benchmark_weight = float(wb[~observable].sum())
        bridge = basket_return - float(row.benchmark_return)
        cost = float(accounting.loc[date, "cost_return"])
        specific_net = contrib["specific_gross"] - cost
        actual = float(row.portfolio_return - row.benchmark_return)
        explained = contrib["industry"] + contrib["style"] + specific_net
        remainder = actual - explained - bridge - unmodelled
        require_close([remainder], [0.0], f"{date}: attribution accounting closure")
        rows.append({"trade_date": date, "exposure_date": lag,
            "portfolio_return": row.portfolio_return, "benchmark_return": row.benchmark_return,
            "excess_return": actual, **contrib, "cost": -cost, "specific_net": specific_net,
            "benchmark_bridge": bridge, "unmodelled_bridge": unmodelled,
            "unobservable_benchmark_weight": unobservable_benchmark_weight,
            "unobservable_attribution_weight_l1": blind_weight,
            "accounting_residual": remainder, "three_block_residual": actual - explained,
            "stale_exposure_l1": stale_l1, "stale_exposure_pnl": stale_pnl})
    output = pd.DataFrame(rows)
    columns = ["industry", "style", "specific_gross", "cost", "specific_net", "benchmark_bridge",
               "unmodelled_bridge", "accounting_residual", "three_block_residual", "stale_exposure_pnl"]
    monthly = linked_monthly(output, columns)
    require_close(monthly[["industry", "style", "specific_net", "benchmark_bridge",
                           "unmodelled_bridge", "accounting_residual"]].sum(axis=1),
                  monthly.excess_return, "monthly attribution linkage")
    annual = {c: float(monthly[c].mean() * 12) for c in columns + ["excess_return"]}
    annual_abs_residual = float(monthly.three_block_residual.abs().mean() * 12)
    per_year = monthly.assign(year=monthly.month.str[:4]).groupby("year").three_block_residual.agg(["mean", lambda x: x.abs().mean()]) * 12
    annual_year_max = float(per_year.iloc[:, 1].max())
    denominator = annual["excess_return"]
    result = {"annualized_contributions": annual,
        "specific_share_net": annual["specific_net"] / denominator if denominator > 0 else None,
        "specific_share_gross": annual["specific_gross"] / denominator if denominator > 0 else None,
        "specific_share_denominator": "D11 monthly mean net excess * 12; undefined when nonpositive",
        "annualized_mean_absolute_three_block_residual": annual_abs_residual,
        "maximum_calendar_year_annualized_absolute_residual": annual_year_max,
        "three_block_residual_passed": bool(annual_abs_residual < .005 and annual_year_max < .005),
        "maximum_stale_exposure_l1": float(output.stale_exposure_l1.max()),
        # Index members Qlib cannot price on a given day. They leave the fit, the
        # contributions and the basket together, so the closure is untouched and
        # their index effect stays inside the official-index bridge -- disclosed
        # here rather than absorbed into specific alpha.
        "days_with_unobservable_constituents": int(output.unobservable_benchmark_weight.gt(0).sum()),
        "maximum_unobservable_benchmark_weight": float(output.unobservable_benchmark_weight.max()),
        "mean_unobservable_benchmark_weight": float(output.unobservable_benchmark_weight.mean()),
        "decision": "No-Go", "decision_changed": False}
    return output, monthly, pd.DataFrame(factors_rows), pd.DataFrame(fits), result

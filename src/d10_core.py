"""D10 ex-post diagnostics. No fitting of alpha models or portfolio optimization.

All accounting is on the frozen, actually executed path. Official-index bridges
remain explicit; they are never relabelled as stock-specific alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.d3_core import CostModel, trade_cost
from src.d9_core import aggregate_daily_to_monthly


class DiagnosticError(ValueError):
    pass


def require_close(actual, expected, label, *, atol=2e-9, rtol=2e-9):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise DiagnosticError(f"{label}: missing/nonfinite values or shape mismatch")
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise DiagnosticError(f"{label}: maximum absolute difference {error:.12g}")
    return error


def unique_frame(frame, keys, label):
    if frame.empty or frame.duplicated(keys).any():
        raise DiagnosticError(f"{label}: empty input or duplicate {keys}")


def price_panels(market):
    """Exactly D3's causal mark rules, with no backward fill.

    Input must start at the ORIGINAL backtest start, not at a warmup date.
    A prior VWAP without a close is not a valid last-close fallback.

    Qlib serves prices as float32, and D3 widens every quote to float64 before
    it computes anything (_coalesce_numeric_series). The panels must widen the
    same way: `mark / execution - 1` and `mark / mark.shift(1) - 1` cancel a
    ratio near 1.0, so in float32 a half-ULP of the quotient (~6e-8) survives
    as roughly 1e-5 of the return itself. Priced against a rebalance notional
    that is a large multiple of NAV, that lands as a ~1e-9 fraction of NAV in
    the intraday PnL, while NAV itself stays exact -- reconciliation against
    the frozen D3 ledger then fails on the derived identity only.
    """
    unique_frame(market, ["trade_date", "instrument"], "market")
    close = market.pivot(index="trade_date", columns="instrument", values="close").sort_index()
    vwap = market.pivot(index="trade_date", columns="instrument", values="vwap").reindex_like(close)
    close, vwap = close.astype("float64"), vwap.astype("float64")
    close, vwap = close.where(close > 0), vwap.where(vwap > 0)
    previous_close = close.ffill().shift(1)
    return (close.combine_first(vwap).combine_first(previous_close),
            vwap.combine_first(close).combine_first(previous_close))


def tradable_fill_range(desired, position, daily_amount, can_buy, can_sell, participation_limit):
    """The closed interval of net fills D3's matcher could have produced.

    D3 gates each LEG, never the sign of the net fill:

        sells          = (-desired.clip(upper=0)).clip(upper=positions)
        executable     = sells.where(can_sell, 0).clip(upper=capacity)
        candidate_buys = desired.clip(lower=0).where(can_buy, 0).clip(upper=capacity)
        executed       = candidate_buys * cash_scale - executable,  cash_scale in [0, 1]

    Clipping the sell leg UP TO the pretrade position is why the sign is not the
    leg. A full exit sells units*price and books back (units*price)/price, which
    in float64 can overshoot; `units.where(units.abs() > 1e-12)` keeps the
    negative residue, and on a later rebalance that name carries a position of
    about -1e-11 CNY. Clipping -0.0 up to a negative position yields a negative
    sell, so `-executable` is a positive, BUY-DIRECTION fill of the dust that the
    SELL leg produced and can_sell allowed -- with can_buy false throughout.
    Reading the leg off the sign calls that a forbidden buy; reconstructing both
    legs prices it exactly, and also bounds capacity and cash scaling.
    """
    capacity = float(daily_amount) * float(participation_limit) if float(daily_amount) > 0 else 0.0
    sell_leg = min(max(-desired, 0.0), position)
    executed_sell = min(sell_leg if can_sell else 0.0, capacity)
    candidate_buy = min(max(desired, 0.0) if can_buy else 0.0, capacity)
    return -executed_sell, -executed_sell + candidate_buy


@dataclass
class ExecutedPath:
    daily: pd.DataFrame
    beginning_weights: pd.DataFrame
    trade_weights: pd.DataFrame
    ending_weights: pd.DataFrame
    pretrade: dict
    orders: pd.DataFrame


def reconstruct_executed_path(daily, trades, market, initial_capital, cost_model):
    """Recover units from logged signed notionals; independently reconcile NAV.

    beginning_weights use previous close NAV; trade_weights use executed VWAP
    notional / previous close NAV. Pretrade weights use execution-time NAV.
    No desired order, including a blocked order, ever creates units or a fee.
    """
    daily = daily.sort_values("trade_date").copy().reset_index(drop=True)
    unique_frame(daily, ["trade_date"], "daily")
    if not trades.empty and trades.duplicated(["execution_date", "instrument"]).any():
        raise DiagnosticError("duplicate executed orders")
    mark, execution = price_panels(market)
    if not pd.DatetimeIndex(daily.trade_date).isin(mark.index).all():
        raise DiagnosticError("daily NAV dates missing from market")
    if not set(trades.execution_date).issubset(set(daily.trade_date)):
        raise DiagnosticError("trades outside saved NAV dates")
    names = mark.columns
    units = pd.Series(0.0, index=names)
    cash = prev_nav = float(initial_capital)
    grouped = {d: g.set_index("instrument") for d, g in trades.groupby("execution_date")}
    rows, begin, changes, end, order_rows = [], [], [], [], []
    pretrade = {}
    previous_marks = mark.shift(1)
    market_days = {d: g.set_index("instrument") for d, g in market.groupby("trade_date")}
    for row in daily.itertuples(index=False):
        date = pd.Timestamp(row.trade_date)
        mp, ep, oldp = mark.loc[date], execution.loc[date], previous_marks.loc[date]
        held = units.abs() > 1e-12
        if mp[held].isna().any() or oldp[held].isna().any() or ep[held].isna().any():
            raise DiagnosticError(f"{date}: missing price for a real holding")
        bw = units * oldp.fillna(0) / prev_nav
        old_pnl = float((units * (mp - oldp).fillna(0)).sum())
        values = units * ep.fillna(0)
        pre_nav = float(cash + values.sum())
        tw = pd.Series(0.0, index=names)
        costs = 0.0
        turnover = 0.0
        g = grouped.get(date)
        if g is not None:
            if g.index.difference(names).size:
                raise DiagnosticError("order instruments missing from market")
            if g.signal_date.nunique() != 1:
                raise DiagnosticError("multiple signals on one execution date")
            signal = pd.Timestamp(g.signal_date.iloc[0])
            live = values / pre_nav
            pretrade[signal] = live[live.abs() > 0].copy()
            pretrade[signal].loc["CASH"] = cash / pre_nav
            require_close(g.pretrade_weight, live.reindex(g.index), f"{date}: logged pretrade weights")
            signed = g.executed_trade_amount.astype(float)
            nonzero = signed.ne(0)
            if ep.reindex(g.index[nonzero]).isna().any():
                raise DiagnosticError("execution price absent for actual trade")
            require_close(g.desired_trade_amount,
                          g.target_weight * pre_nav - values.reindex(g.index), f"{date}: desired notional", atol=1e-5)
            # The pretrade-weight check above already bounds how far a rebuilt
            # position may sit from D3's own; reuse that budget, do not invent one.
            position_budget = max(1e-5, 2e-9 * pre_nav)
            for name, order in g.iterrows():
                quote = market_days[date].loc[name]
                calculated = trade_cost(order.executed_trade_amount, quote.volatility,
                                        quote.daily_amount_cny, date, cost_model)
                max_error = max(require_close([order[k]], [v], f"{date}/{name}/{k}", atol=1e-5)
                                for k, v in calculated.items())
                signed_amount = float(order.executed_trade_amount)
                position = float(values.get(name, 0.0))
                low, high = tradable_fill_range(float(order.desired_trade_amount), position,
                                                float(quote.daily_amount_cny), bool(quote.can_buy),
                                                bool(quote.can_sell), cost_model.participation_limit)
                if not low - position_budget <= signed_amount <= high + position_budget:
                    raise DiagnosticError(
                        f"{date}/{name}: fill {signed_amount:.12g} outside the tradable leg range "
                        f"[{low:.12g}, {high:.12g}]; can_buy={bool(quote.can_buy)} "
                        f"can_sell={bool(quote.can_sell)} pretrade position {position:.12g}")
                desired_amount = float(order.desired_trade_amount)
                side_allowed = (bool(quote.can_buy) if desired_amount > 0
                                else bool(quote.can_sell) if desired_amount < 0 else True)
                if bool(order.blocked_by_tradability) != bool(desired_amount and not side_allowed):
                    raise DiagnosticError(f"{date}/{name}: saved blocked_by_tradability disagrees with input")
                expected_buy = bool(quote.can_buy) and pd.notna(ep.loc[name])
                expected_sell = bool(quote.can_sell) and pd.notna(ep.loc[name])
                if bool(order.can_buy) != expected_buy or bool(order.can_sell) != expected_sell:
                    raise DiagnosticError(f"{date}/{name}: saved tradability disagrees with input")
                order_rows.append({"execution_date": date, "signal_date": signal, "instrument": name,
                    "executed_trade_amount": signed_amount, "desired_trade_amount": desired_amount,
                    "total_cost": float(order.total_cost), "cost_reconciliation_error_cny": max_error,
                    "blocked_by_tradability": bool(order.blocked_by_tradability),
                    "participation_limited": bool(order.participation_limited),
                    "cash_limited": bool(order.cash_limited),
                    "pretrade_position_cny": position,
                    "negative_position_dust_fill": bool(position < 0 and signed_amount != 0),
                    "post_execution_pnl_cny": float(signed_amount * (mp.loc[name] / ep.loc[name] - 1))
                        if signed_amount else 0.0})
            delta = signed[nonzero] / ep.reindex(signed.index[nonzero])
            units.loc[delta.index] += delta
            units = units.where(units.abs() > 1e-12, 0.0)
            tw.loc[signed.index] = signed / prev_nav
            costs = float(g.total_cost.sum())
            cash -= float(signed.sum()) + costs
            # Same representational tolerance as D3; no economic cash injection.
            if -max(1e-8, pre_nav * 1e-12) <= cash < 0:
                cash = 0.0
            if cash < 0 or (units < -1e-10).any():
                raise DiagnosticError("negative reconstructed cash or units")
            turnover = 0.5 * float(signed.abs().sum()) / pre_nav
        trade_pnl = float((tw * (mp / ep - 1).fillna(0)).sum()) * prev_nav
        nav = float(cash + (units * mp.fillna(0)).sum())
        nav_error = require_close([nav], [row.nav], f"{date}: NAV", atol=1e-4)
        require_close([cash], [row.cash], f"{date}: cash", atol=1e-4)
        require_close([costs], [row.transaction_cost], f"{date}: daily costs", atol=1e-4)
        require_close([turnover], [row.single_sided_turnover], f"{date}: executed turnover")
        actual_return = float(row.nav / prev_nav - 1)
        require_close([actual_return], [row.portfolio_return], f"{date}: saved daily return")
        pnl_return = (old_pnl + trade_pnl - costs) / prev_nav
        require_close([pnl_return], [actual_return], f"{date}: holding + intraday - costs")
        begin.append(bw.to_numpy()); changes.append(tw.to_numpy())
        end.append((units * mp.fillna(0) / nav).to_numpy())
        rows.append({"trade_date": date, "previous_nav": prev_nav,
                     "pretrade_nav": pre_nav, "holding_return": old_pnl / prev_nav,
                     "execution_return": trade_pnl / prev_nav, "cost_return": costs / prev_nav,
                     "accounting_residual": actual_return - pnl_return,
                     "nav_error_cny": nav_error, "cash_weight": cash / nav})
        prev_nav = float(row.nav)
    dates = pd.DatetimeIndex(daily.trade_date, name="trade_date")
    panel = lambda values: pd.DataFrame(values, index=dates, columns=names)
    return ExecutedPath(pd.DataFrame(rows), panel(begin), panel(changes), panel(end),
                        pretrade, pd.DataFrame(order_rows))


def half_l1(left, right):
    names = left.index.union(right.index)
    return float(0.5 * (left.reindex(names, fill_value=0) - right.reindex(names, fill_value=0)).abs().sum())


def turnover_split(target, reference, members):
    """V2's fully invested forced-exit PLUS replacement convention."""
    total = half_l1(target, reference)
    mandatory = float(reference.loc[~reference.index.isin(members)].sum())
    if mandatory > total + 2e-7:
        raise DiagnosticError("mandatory turnover exceeds total; reference must be fully invested")
    return {"target_total_turnover": total, "target_mandatory_turnover": mandatory,
            "target_discretionary_turnover": max(total - mandatory, 0.0)}


def executed_turnover_split(orders, members, pretrade_nav):
    """An exact, explicitly leg-based split; replacement buys are not identifiable.

    Report mandatory SELL HALF-LEG, and the other half-legs. Do not compare
    this number directly with V2's forced-exit-plus-replacement target number.
    """
    signed = orders.set_index("instrument").executed_trade_amount
    sells = -signed.clip(upper=0)
    forced = float(sells.loc[~sells.index.isin(members)].sum())
    total = float(signed.abs().sum() * 0.5 / pretrade_nav)
    mandatory_leg = 0.5 * forced / pretrade_nav
    return {"executed_total_turnover": total, "executed_mandatory_sell_leg": mandatory_leg,
            "executed_other_legs": total - mandatory_leg,
            "executed_forced_sell_weight": forced / pretrade_nav,
            "executed_buy_weight": float(signed.clip(lower=0).sum() / pretrade_nav),
            "executed_sell_weight": float(sells.sum() / pretrade_nav),
            "desired_live_turnover": float(orders.desired_trade_amount.abs().sum() * 0.5 / pretrade_nav),
            "execution_shortfall_turnover": float((orders.desired_trade_amount - orders.executed_trade_amount).abs().sum() * 0.5 / pretrade_nav)}


def path_gross_net(daily, initial_capital):
    """Cost add-back on the realized holdings path (not a costless rerun)."""
    daily = daily.sort_values("trade_date")
    previous = daily.nav.shift(1).fillna(initial_capital)
    cost = daily.transaction_cost / previous
    gross = daily.portfolio_return + cost
    if not np.isfinite(gross).all() or (gross <= -1).any() or len(daily) < 2:
        raise DiagnosticError("invalid gross/net return input")
    exponent = 252 / (len(daily) - 1)  # preserved D8 ladder convention
    net_ann = float(np.prod(1 + daily.portfolio_return) ** exponent - 1)
    gross_ann = float(np.prod(1 + gross) ** exponent - 1)
    bench_ann = float(np.prod(1 + daily.benchmark_return) ** exponent - 1)
    gm = daily[["trade_date", "benchmark_return"]].assign(portfolio_return=gross)
    monthly = aggregate_daily_to_monthly(daily)
    gross_monthly = aggregate_daily_to_monthly(gm)
    return {"net_excess_cagr_difference": net_ann - bench_ann,
            "gross_excess_cagr_difference": gross_ann - bench_ann,
            "cost_drag_cagr_difference": gross_ann - net_ann,
            "net_excess_monthly_annualized": float(monthly.excess_return.mean() * 12),
            "gross_excess_monthly_annualized": float(gross_monthly.excess_return.mean() * 12),
            "cost_drag_monthly_annualized": float((gross_monthly.excess_return - monthly.excess_return).mean() * 12),
            "transaction_cost_cny": float(daily.transaction_cost.sum())}


def wls_cross_section(exposure, returns, industries, styles):
    """min sum sqrt(lagged cap) * (r - X f)^2, NO intercept or ridge.

    Full industry dummies encode the common market return. Zero industry
    columns are not estimable that day; all six style columns remain required.
    """
    if exposure.index.duplicated().any():
        raise DiagnosticError("duplicate WLS stocks")
    active_industries = [c for c in industries if exposure[c].abs().sum() > 0]
    columns = active_industries + list(styles)
    x = exposure[columns].to_numpy(float)
    y = returns.reindex(exposure.index).to_numpy(float)
    cap = exposure.market_cap.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(cap).all() or (cap <= 0).any():
        raise DiagnosticError("invalid WLS data; no silent stock deletion")
    scale = (cap / np.median(cap)) ** 0.25  # sqrt of objective weights
    coefficients, _, rank, singular = np.linalg.lstsq(x * scale[:, None], y * scale, rcond=None)
    if rank != len(columns) or len(y) <= rank:
        raise DiagnosticError("rank-deficient WLS; no fitted fallback")
    factors = pd.Series(0.0, index=list(industries) + list(styles))
    factors.loc[columns] = coefficients
    residual = y - x @ coefficients
    objective_weights = scale ** 2
    mean = float(np.average(y, weights=objective_weights))
    tss = float(np.sum(objective_weights * (y - mean) ** 2))
    audit = {"observations": len(y), "rank": int(rank), "effective_columns": len(columns),
             "condition_number": float(singular[0] / singular[-1]),
             "weighted_r_squared": 1 - float(np.sum(objective_weights * residual ** 2)) / tss if tss > 0 else None,
             "max_weighted_normal_equation_error": float(np.max(np.abs(x.T @ (objective_weights * residual))))}
    return factors, audit


def linked_monthly(daily, component_columns):
    """Exact telescoping link to compound(Rp) - compound(Rb) each month.

    a_t = product_{s<t}(1+Rp_s) product_{s>t}(1+Rb_s).
    Daily components (including bridges) use the SAME link coefficient.
    """
    unique_frame(daily, ["trade_date"], "linking")
    rows = []
    for month, g in daily.sort_values("trade_date").groupby(pd.to_datetime(daily.sort_values("trade_date").trade_date).dt.to_period("M")):
        rp, rb = g.portfolio_return.to_numpy(float), g.benchmark_return.to_numpy(float)
        values = g[list(component_columns)].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(rp).all() or not np.isfinite(rb).all() or (rp <= -1).any() or (rb <= -1).any():
            raise DiagnosticError("nonfinite/invalid monthly linking input")
        before = np.r_[1.0, np.cumprod(1 + rp)[:-1]]
        after = np.r_[np.cumprod((1 + rb)[::-1])[::-1][1:], 1.0]
        rp_m, rb_m = float(np.prod(1 + rp) - 1), float(np.prod(1 + rb) - 1)
        row = {"month": str(month), "month_end": g.trade_date.max(), "portfolio_return": rp_m,
               "benchmark_return": rb_m, "excess_return": rp_m - rb_m,
               **dict(zip(component_columns, (values * (before * after)[:, None]).sum(axis=0)))}
        rows.append(row)
    return pd.DataFrame(rows)


def attribute_executed_path(daily, path, market, exposures, industries, styles):
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
    mark, execution = price_panels(market)
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
        stale_l1, stale_pnl, unmodelled = 0.0, 0.0, 0.0
        for period, returns, weights in [("close_to_close", cc.loc[date], active),
                                          ("execution_to_close", intraday.loc[date], tw)]:
            if period == "execution_to_close" and not (weights.abs() > 1e-14).any():
                continue
            f, audit = wls_cross_section(fit, returns, industries, styles)
            fits.append({"trade_date": date, "exposure_date": lag, "period": period, **audit})
            factors_rows.extend({"trade_date": date, "exposure_date": lag, "period": period,
                                 "factor": name, "factor_return": float(value)} for name, value in f.items())
            w = weights.reindex(identified, fill_value=0)
            x = latest.loc[identified, factor_cols].astype(float)
            r = returns.reindex(identified)
            if r[w.abs() > 1e-14].isna().any():
                raise DiagnosticError(f"{date}: missing realized attribution return")
            r = r.fillna(0)
            industry_r = x[list(industries)] @ f[list(industries)]
            style_r = x[list(styles)] @ f[list(styles)]
            contrib["industry"] += float(w @ industry_r)
            contrib["style"] += float(w @ style_r)
            contrib["specific_gross"] += float(w @ (r - industry_r - style_r))
            stale = pd.to_datetime(latest.loc[identified, "exposure_date"]).lt(lag)
            stale_l1 += float(w[stale].abs().sum()); stale_pnl += float(w[stale] @ r[stale])
            uw = weights.reindex(unseen, fill_value=0)
            ur = returns.reindex(unseen)
            if ur[uw.abs() > 1e-14].isna().any():
                raise DiagnosticError("unmodelled holding also lacks realized price")
            unmodelled += float(uw @ ur.fillna(0))
        basket_return = float(wb @ cc.loc[date].reindex(wb.index))
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
        "specific_share_denominator": "D9 monthly mean net excess * 12; undefined when nonpositive",
        "annualized_mean_absolute_three_block_residual": annual_abs_residual,
        "maximum_calendar_year_annualized_absolute_residual": annual_year_max,
        "three_block_residual_passed": bool(annual_abs_residual < .005 and annual_year_max < .005),
        "maximum_stale_exposure_l1": float(output.stale_exposure_l1.max()),
        "decision": "No-Go", "decision_changed": False}
    return output, monthly, pd.DataFrame(factors_rows), pd.DataFrame(fits), result


def stress_schedule(targets, calendar, scenario):
    """Frozen-target execution sensitivity, never an optimized new strategy."""
    if scenario not in {"delay_one_day", "biweekly"}:
        raise DiagnosticError("unknown fixed stress scenario")
    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    signals = pd.DatetimeIndex(targets.signal_date.unique()).normalize().sort_values()
    if signals.empty or not signals.isin(calendar).all():
        raise DiagnosticError("stress signals missing from trading calendar")
    anchor = signals[0].to_period("W-FRI").ordinal
    rows = []
    for signal in signals:
        selected = scenario != "biweekly" or (signal.to_period("W-FRI").ordinal - anchor) % 2 == 0
        delay = 2 if scenario == "delay_one_day" else 1
        index = calendar.get_loc(signal) + delay
        valid = index < len(calendar)
        rows.append({"source_signal_date": signal, "selected": selected,
                     "status": "scheduled" if selected and valid else "alternate_week_omitted" if not selected else "beyond_cutoff",
                     "dispatch_date": calendar[index - 1] if selected and valid else pd.NaT,
                     "execution_date": calendar[index] if selected and valid else pd.NaT})
    schedule = pd.DataFrame(rows)
    selected = schedule.loc[schedule.status.eq("scheduled")]
    dispatched = targets.drop(columns=["execution_date"], errors="ignore").rename(columns={"signal_date": "source_signal_date"}).merge(
        selected, on="source_signal_date", validate="many_to_one")
    dispatched["signal_date"] = dispatched.dispatch_date
    return dispatched, schedule


def align_stress_daily(nav, reference_daily, initial_capital):
    """Use main's entire date range, including cash while a delayed trade waits."""
    unique_frame(nav, ["trade_date"], "stress NAV")
    ref = reference_daily.sort_values("trade_date")
    result = nav.set_index("trade_date").reindex(pd.DatetimeIndex(ref.trade_date)).copy()
    if result.nav.isna().any():
        raise DiagnosticError("stress has missing main-calendar dates")
    result["portfolio_return"] = result.nav.pct_change(fill_method=None)
    result.iloc[0, result.columns.get_loc("portfolio_return")] = result.nav.iloc[0] / initial_capital - 1
    result["benchmark_return"] = ref.benchmark_return.to_numpy(float)
    result["excess_return"] = result.portfolio_return - result.benchmark_return
    return result.rename_axis("trade_date").reset_index()

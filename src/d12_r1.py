"""R1 simulator: D11's causal path with one registered decision-side impact term.

The execution path, fees, order freezing and cash feedback remain the protected D11
implementation.  With decision ``k=0`` this module delegates structurally to D11's
simulator, so no unused power cone, estimate fallback or amount rule can perturb R0.
For non-zero ``k`` only the target objective changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.d3_core import CostModel
from src.d7_core import OptimizationLimits, PortfolioOptimizationError
from src.d11_core import (CausalityError, execute_orders, freeze_orders,
                          optimize_at_signal, signal_eligibility, simulate_causal, unique)
from src.d12_core import (ImpactEstimator, coefficients_at, expected_impact_fraction,
                          impact_penalty, rolling_estimates)
from src.d12_repo import D12Error


@dataclass
class R1Simulation:
    nav: pd.DataFrame
    targets: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    audits: pd.DataFrame


def _finite_quantiles(values):
    data = pd.to_numeric(pd.Series(values), errors="coerce")
    data = data.loc[np.isfinite(data)]
    if data.empty:
        return {"minimum": None, "median": None, "p99": None, "maximum": None}
    return {"minimum": float(data.min()), "median": float(data.median()),
            "p99": float(data.quantile(.99)), "maximum": float(data.max())}


def optimize_at_signal_r1(mu, benchmark, pretrade, exposures, buy_allowed,
                          industries, styles, limits, costs, execution_date,
                          coefficients, amount_buy_forbidden, *, initial=False,
                          estimator=None):
    """D11 target optimisation plus the frozen convex expected-impact penalty."""
    if estimator is not None and estimator.bypassed:
        return optimize_at_signal(mu, benchmark, pretrade, exposures, buy_allowed,
                                  industries, styles, limits, costs, execution_date,
                                  initial=initial)
    import cvxpy as cp

    limits.validate()
    idx = benchmark.index
    if idx.has_duplicates or pretrade.index.has_duplicates:
        raise CausalityError("Duplicate portfolio instruments")
    wb, w0 = benchmark.astype(float), pretrade.astype(float)
    factors = exposures.reindex(idx)[list(industries) + list(styles)].to_numpy(float)
    coefficient = pd.to_numeric(pd.Series(coefficients).reindex(idx), errors="coerce")
    amount_block = pd.Series(amount_buy_forbidden).reindex(idx).fillna(True).astype(bool)
    if (not np.isfinite(wb).all() or not np.isfinite(w0).all()
            or not np.isfinite(mu.reindex(idx)).all() or not np.isfinite(factors).all()
            or not np.isfinite(coefficient).all() or coefficient.lt(0).any()
            or wb.lt(0).any() or w0.lt(0).any()
            or not np.isclose(wb.sum(), 1, atol=1e-8, rtol=0)
            or w0.sum() > 1 + 1e-8):
        raise CausalityError("Invalid R1 optimization inputs")
    if initial and w0.sum() > 1e-10:
        raise CausalityError("Initial exemption requires cash-only portfolio")

    old = w0.reindex(idx, fill_value=0.0).to_numpy()
    outside = w0.loc[~w0.index.isin(idx)]
    forced = float(outside.sum())
    outside_coefficient = pd.to_numeric(pd.Series(coefficients).reindex(outside.index), errors="coerce")
    if len(outside) and (not np.isfinite(outside_coefficient).all() or outside_coefficient.lt(0).any()):
        raise CausalityError("Forced exits need finite non-negative impact coefficients")

    w = cp.Variable(len(idx))
    active = w - wb.to_numpy()
    total_l1 = cp.norm1(w - old) + forced
    constraints = [cp.sum(w) == 1, w >= 0,
                   active <= limits.stock_active, active >= -limits.stock_active]
    for columns, bound in [(industries, limits.industry_active), (styles, limits.style_active)]:
        if columns:
            factor_active = exposures.reindex(idx)[list(columns)].to_numpy(float).T @ active
            constraints += [factor_active <= bound, factor_active >= -bound]
    allowed = buy_allowed.reindex(idx).eq(True).to_numpy() & ~amount_block.to_numpy()
    if (~allowed).any():
        constraints.append(w[~allowed] <= old[~allowed])
    if not initial:
        constraints.append(.5 * total_l1 - forced <= limits.single_sided_turnover)

    buys = cp.sum(cp.pos(w - old))
    sells = cp.sum(cp.pos(old - w)) + forced
    linear_fee = (costs.commission_rate + costs.transfer_fee_rate) * (buys + sells)
    linear_fee += costs.stamp_rate(execution_date) * sells
    variable_impact = impact_penalty(w, old, coefficient.to_numpy(), mode="abs")
    forced_impact = float((outside_coefficient * outside.abs().pow(1.5)).sum())
    problem = cp.Problem(cp.Maximize(mu.reindex(idx).to_numpy() @ active
                                     - linear_fee - variable_impact - forced_impact), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status != "optimal" or w.value is None:
        raise PortfolioOptimizationError(f"D12 R1 strict solver status={problem.status}; no fallback")

    target = pd.Series(np.maximum(w.value, 0.0), index=idx)
    target.loc[~allowed] = np.minimum(target.loc[~allowed], old[~allowed])
    active_value = target - wb
    turnover = .5 * (float((target - old).abs().sum()) + forced)
    changes = pd.concat([(target - pd.Series(old, index=idx)), -outside])
    impact = expected_impact_fraction(changes, pd.Series(coefficients).reindex(changes.index))
    buy_weight = float((target - pd.Series(old, index=idx)).clip(lower=0).sum())
    sell_weight = float((pd.Series(old, index=idx) - target).clip(lower=0).sum() + forced)
    linear_fee_value = ((costs.commission_rate + costs.transfer_fee_rate) * (buy_weight + sell_weight)
                        + costs.stamp_rate(execution_date) * sell_weight)
    audit = {
        "solver_status": problem.status, "fallback_used": False,
        "initial_construction_exempt": bool(initial),
        "pretrade_cash_weight": float(1 - w0.sum()),
        "maximum_stock_active": float(active_value.abs().max()),
        "maximum_industry_active": float((exposures.reindex(idx)[list(industries)].T @ active_value).abs().max()) if industries else 0.,
        "maximum_style_active": float((exposures.reindex(idx)[list(styles)].T @ active_value).abs().max()) if styles else 0.,
        "target_total_turnover": turnover, "mandatory_turnover": forced,
        "target_active_turnover": max(turnover - forced, 0.),
        "target_weight_sum_error": abs(float(target.sum()) - 1),
        "prohibited_buy_weight": float((target.loc[~allowed] - pd.Series(old, index=idx).loc[~allowed]).clip(lower=0).sum()),
        "amount_buy_forbidden_names": int(amount_block.sum()),
        "expected_impact_fraction": impact,
        "expected_impact_variable_fraction": float(variable_impact.value),
        "expected_impact_forced_fraction": forced_impact,
        "objective_expected_excess_fraction": float(mu.reindex(idx).to_numpy() @ active_value.to_numpy()),
        "objective_linear_fee_fraction": float(linear_fee_value),
        "objective_net_fraction": float(problem.value),
    }
    checks = [audit["target_weight_sum_error"] <= limits.tolerance,
              audit["maximum_stock_active"] <= limits.stock_active + limits.tolerance,
              audit["maximum_industry_active"] <= limits.industry_active + limits.tolerance,
              audit["maximum_style_active"] <= limits.style_active + limits.tolerance,
              audit["prohibited_buy_weight"] <= limits.tolerance,
              initial or audit["target_active_turnover"] <= limits.single_sided_turnover + limits.tolerance,
              math.isclose(audit["expected_impact_fraction"],
                           audit["expected_impact_variable_fraction"] + forced_impact,
                           rel_tol=1e-8, abs_tol=1e-12)]
    if not all(checks):
        raise PortfolioOptimizationError(f"D12 R1 target audit failed: {audit}")
    audit["passed"] = True
    return target, audit


def annotate_orders(orders, estimates, signal_date, coefficients, amount_buy_forbidden,
                    *, participation_limit=.05):
    """Attach pre-registered, decision-time participation and impact telemetry."""
    if orders.empty:
        return orders.copy()
    signal_date = pd.Timestamp(signal_date).normalize()
    same_day = estimates.loc[estimates.trade_date.eq(signal_date)].set_index("instrument")
    out = orders.set_index("instrument").copy()
    out["decision_coefficient"] = pd.Series(coefficients).reindex(out.index)
    out["amount_estimate_cny"] = same_day.a_est.reindex(out.index)
    out["sigma_est"] = same_day.sigma_est.reindex(out.index)
    out["sigma_source"] = same_day.sigma_source.reindex(out.index)
    out["amount_buy_forbidden"] = pd.Series(amount_buy_forbidden).reindex(out.index).fillna(True).astype(bool)
    out["order_weight"] = out.order_units * out.signal_price / out.signal_nav
    out["expected_impact_fraction"] = out.decision_coefficient * out.order_weight.abs().pow(1.5)
    valid_amount = pd.to_numeric(out.amount_estimate_cny, errors="coerce")
    valid_amount = valid_amount.where(np.isfinite(valid_amount) & valid_amount.gt(0))
    out["ex_ante_participation"] = out.order_weight.abs() * out.signal_nav / valid_amount
    out["ex_ante_above_limit"] = out.ex_ante_participation.gt(float(participation_limit))
    out["side"] = np.where(out.order_units.gt(0), "buy", "sell")
    if out.expected_impact_fraction.isna().any():
        raise D12Error(f"{signal_date.date()}: an order has no decision-side impact coefficient")
    if (out.loc[out.side == "buy", "amount_buy_forbidden"]).any():
        raise D12Error(f"{signal_date.date()}: submitted a buy whose amount estimate was unavailable")
    return out.reset_index()


def simulate_r1(market, predictions, exposures, flags, calendar, config,
                industries, styles, estimator, *, progress=None):
    """Run the full causal path; ``k=0`` is an explicit D11 structural bypass."""
    estimator.validate()
    if estimator.bypassed:
        result = simulate_causal(market, predictions, exposures, flags, calendar, config,
                                 industries, styles, progress=progress)
        return R1Simulation(result.nav, result.targets, result.orders, result.trades, result.audits)

    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    unique(market, ["trade_date", "instrument"], "market")
    unique(predictions, ["sample_date", "instrument"], "predictions")
    unique(exposures, ["trade_date", "instrument"], "exposures")
    estimates = rolling_estimates(market, estimator)
    d7 = config["d7"]
    limits = OptimizationLimits(d7["stock_active_limit"], d7["industry_active_limit"],
                                d7["style_active_limit"], d7["single_sided_turnover_limit"],
                                d7["constraint_tolerance"])
    costs = CostModel.from_config(config)
    cash = float(config["d8"]["main_capital_cny"])
    units, marks = pd.Series(dtype=float), pd.Series(dtype=float)
    pending = pd.DataFrame()
    targets, orders, trades, audits, navs = [], [], [], [], []
    price_days = {pd.Timestamp(key): value.set_index("instrument") for key, value in market.groupby("trade_date")}
    pred_days = {pd.Timestamp(key): value.set_index("instrument") for key, value in predictions.groupby("sample_date")}
    exposure_days = {pd.Timestamp(key): value.set_index("instrument").sort_index()
                     for key, value in exposures.groupby("trade_date")}
    start, end = pd.Timestamp(config["windows"]["oos_start"]), pd.Timestamp(config["windows"]["oos_end"])
    dates = calendar[(calendar >= start) & (calendar <= end)]

    for day_number, date in enumerate(dates):
        day = price_days.get(date)
        if day is None:
            raise CausalityError(f"No market session {date}")
        fee, turnover, signal_executed = 0., 0., None
        if not pending.empty:
            if not pending.execution_date.eq(date).all():
                raise CausalityError("Missed pending execution")
            units, cash, fills = execute_orders(units, cash, pending, day, date, costs)
            trades.append(fills)
            fee = float(fills.total_cost.sum())
            turnover = .5 * float(fills.executed_trade_amount.abs().sum()) / float(pending.signal_nav.iloc[0])
            signal_executed = pending.signal_date.iloc[0]
            pending = pd.DataFrame()
        close = pd.to_numeric(day.close, errors="coerce").where(lambda value: (value > 0) & np.isfinite(value)).dropna()
        marks = marks.reindex(marks.index.union(close.index))
        marks.loc[close.index] = close
        if marks.reindex(units.index).isna().any():
            raise CausalityError("Unpriceable actual holding")
        value = float((units * marks.reindex(units.index)).sum())
        nav = cash + value
        if not np.isfinite(nav) or nav <= 0:
            raise CausalityError("Invalid NAV")
        navs.append({"trade_date": date, "nav": nav, "cash": cash, "gross_exposure": value / nav,
                     "transaction_cost": fee, "single_sided_turnover": turnover,
                     "signal_date": signal_executed})
        if date not in pred_days or day_number + 1 == len(dates):
            continue

        next_date = dates[day_number + 1]
        if date not in exposure_days:
            raise CausalityError(f"Missing signal exposures {date}")
        cross = exposure_days[date]
        prediction = pred_days[date]
        if set(prediction.index) != set(cross.index):
            raise CausalityError("Inference/exposure universe mismatch")
        benchmark = cross.benchmark_weight.astype(float)
        benchmark = benchmark / benchmark.sum()
        pretrade = (units * marks.reindex(units.index)) / nav
        mu = prediction[config["d5"]["expected_return_column"]].reindex(cross.index)
        policy = signal_eligibility(flags, date, cross.index, calendar)
        known_price = marks.reindex(cross.index).gt(0)
        all_names = cross.index.union(pretrade.index)
        coefficient, amount_block, estimate_audit = coefficients_at(
            estimates, date, nav, estimator, instruments=all_names)
        allowed = policy.signal_can_buy & known_price & ~amount_block.reindex(cross.index).fillna(True)
        target, audit = optimize_at_signal_r1(
            mu, benchmark, pretrade, cross, allowed, industries, styles, limits, costs,
            next_date, coefficient, amount_block, initial=len(audits) == 0, estimator=estimator)
        pending = freeze_orders(target, units, cash, marks, allowed, date, next_date)
        pending = annotate_orders(pending, estimates, date, coefficient, amount_block,
                                  participation_limit=costs.participation_limit)
        target_rows = pd.DataFrame({
            "instrument": cross.index, "signal_date": date, "execution_date": next_date,
            "target_weight": target.to_numpy(), "benchmark_weight": benchmark.to_numpy(),
            "pretrade_weight": pretrade.reindex(cross.index, fill_value=0).to_numpy(),
            "expected_excess": mu.to_numpy(), "signal_can_buy": allowed.to_numpy(),
            "eligibility_asof": policy.eligibility_asof.to_numpy(), "decision_asof": date})
        targets.append(target_rows)
        orders.append(pending.copy())
        q = _finite_quantiles(coefficient)
        audit.update(
            signal_date=date, execution_date=next_date, decision_asof=date,
            eligibility_asof=policy.eligibility_asof.iloc[0], signal_nav=nav,
            actual_holdings=len(units), prohibited_or_unknown_buy_names=int((~allowed).sum()),
            decision_impact_k=float(estimator.k), estimate_amount_unavailable=estimate_audit["amount_unavailable"],
            estimate_substitute_coefficient=estimate_audit["substitute_coefficient"],
            sigma_source_rolling=int(estimate_audit["sigma_source_counts"].get("rolling", 0)),
            sigma_source_cross_section_median=int(estimate_audit["sigma_source_counts"].get("cross_section_median", 0)),
            sigma_source_constant_fallback=int(estimate_audit["sigma_source_counts"].get("constant_fallback", 0)),
            coefficient_minimum=q["minimum"], coefficient_median=q["median"],
            coefficient_p99=q["p99"], coefficient_maximum=q["maximum"],
            order_rows=int(len(pending)), buy_order_rows=int(pending.order_units.gt(0).sum()),
            sell_order_rows=int(pending.order_units.lt(0).sum()),
            ex_ante_above_5pct_rows=int(pending.ex_ante_above_limit.sum()),
            maximum_ex_ante_participation=float(pending.ex_ante_participation.max()) if pending.ex_ante_participation.notna().any() else None,
            order_expected_impact_fraction=float(pending.expected_impact_fraction.sum()))
        audits.append(audit)
        if progress is not None:
            progress(target_rows, pending, audit)

    combine = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return R1Simulation(pd.DataFrame(navs), combine(targets), combine(orders),
                        combine(trades), pd.DataFrame(audits))


def participation_summary(orders, trades):
    """The registered buy/sell monitoring, with decision and execution sides separate."""
    rows = []
    for side in ["buy", "sell", "all"]:
        order_part = orders if side == "all" else orders.loc[orders.side.eq(side)]
        trade_part = trades if side == "all" else trades.loc[trades.side.eq(side)]
        limited = trade_part.participation_limited.eq(True) if len(trade_part) else pd.Series(dtype=bool)
        unfilled_cny = (trade_part.unfilled_units.abs() * trade_part.signal_price) if len(trade_part) else pd.Series(dtype=float)
        rows.append({
            "side": side, "order_rows": int(len(order_part)),
            "requested_signal_cny": float((order_part.order_units.abs() * order_part.signal_price).sum()),
            "a_est_missing_rows": int(order_part.amount_estimate_cny.isna().sum()),
            "ex_ante_above_5pct_rows": int(order_part.ex_ante_above_limit.sum()),
            "maximum_ex_ante_participation": float(order_part.ex_ante_participation.max()) if order_part.ex_ante_participation.notna().any() else None,
            "expected_impact_fraction": float(order_part.expected_impact_fraction.sum()),
            "actual_capacity_limited_rows": int(limited.sum()),
            "capacity_limited_unfilled_signal_cny": float(unfilled_cny.loc[limited].sum()) if len(trade_part) else 0.,
            "all_unfilled_signal_cny": float(unfilled_cny.sum()) if len(trade_part) else 0.,
            "cash_scaled_rows": int((trade_part.side.eq("buy") & trade_part.cash_scale.lt(1 - 1e-12)
                                      & trade_part.executed_units.ne(0)).sum()) if len(trade_part) else 0,
        })
    return pd.DataFrame(rows)

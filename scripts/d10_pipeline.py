"""D10 diagnostic runner: exclusively reads D1-D9 and writes reports/d10/<id>."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402
from src.d3_core import CostModel, make_label_windows, run_vwap_backtest  # noqa: E402
from src.d6_core import compute_raw_style_history, tushare_to_qlib  # noqa: E402
from src.d8_core import make_ideal_execution_market  # noqa: E402
from src.d9_core import aggregate_daily_to_monthly, summarize_monthly_performance  # noqa: E402
from src.d10_core import (  # noqa: E402
    DiagnosticError, align_stress_daily, attribute_executed_path, executed_turnover_split,
    half_l1, linked_monthly, path_gross_net, price_panels, reconstruct_executed_path,
    require_close, stress_schedule, turnover_split, unique_frame,
)
from src.d10_io import (  # noqa: E402
    D10Output, assert_snapshot_unchanged, preflight, snapshot_inputs, verify_repository,
)
from scripts.d3_pipeline import (  # noqa: E402
    add_ex_ante_volatility, add_exchange_flags, load_qlib_market, prepare_markets,
)
from scripts.d6_pipeline import assemble_exposures  # noqa: E402
from scripts.d7_pipeline import build_execution_price_matrix  # noqa: E402
from scripts.d8_pipeline import build_frozen_d7_pretrade_path  # noqa: E402


def read_frame(access, path, date_col, start, end):
    frame = access.read_parquet(path, date_col=date_col, start_date=start, end_date=end)
    for col in [date_col, "trade_date", "sample_date", "signal_date", "execution_date"]:
        if col in frame:
            frame[col] = parse_date_series(frame[col])
    if frame.empty:
        raise DiagnosticError(f"Empty required diagnostic input: {path}")
    return frame


def read_years(access, config, endpoint, start, end):
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = Path(config["data"]["tushare_raw_dir"]) / f"{endpoint}_{year}.parquet"
        if not (access.root / path).exists():
            raise DiagnosticError(f"Missing historical input: {path}")
        frame = access.read_parquet(path, date_col="trade_date", start_date=start, end_date=end)
        frame["trade_date"] = parse_date_series(frame.trade_date)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_window(access, config, window, output):
    start = pd.Timestamp(config["windows"][f"{window}_start"])
    end = pd.Timestamp(config["windows"][f"{window}_end"])
    prefix = "d8" if window == "design" else "d9_v2"
    reports = Path(config["data"]["reports_dir"])
    processed = Path(config["data"]["processed_design_dir"] if window == "design"
                     else config["d9"]["v2"]["processed_oos_dir"])
    weights = read_frame(access, processed / "benchmark_weights.parquet", "trade_date", start, end)
    flags = read_frame(access, processed / "tradability_flags.parquet", "trade_date", start, end)
    official = read_frame(access, Path(config["data"]["tushare_raw_dir"]) / "index_daily_H00905_CSI.parquet",
                          "trade_date", start, end).sort_values("trade_date")
    unique_frame(official, ["trade_date"], "official benchmark")
    official["benchmark_return"] = official.close.pct_change(fill_method=None)
    instruments = sorted(weights.instrument.unique())
    history_start = start if window == "design" else start - pd.Timedelta(days=int(config["d6"]["history_buffer_calendar_days"]))
    print(f"D10 {window}: read frozen price/state inputs", flush=True)
    market = load_qlib_market(access, instruments, history_start, end,
                             float(config["tradability"]["qlib_amount_multiplier_to_cny"]))
    limits = read_years(access, config, "stk_limit", start, end)
    suspensions = read_years(access, config, "suspend_d", start, end)
    market = add_exchange_flags(market, limits, suspensions, config)
    market = add_ex_ante_volatility(market, config)
    strategy, _ = prepare_markets(market, flags)
    strategy = strategy.loc[strategy.trade_date.between(start, end)].copy()
    ideal = make_ideal_execution_market(strategy, fallback_daily_amount=float(config["tradability"]["min_avg_daily_amount_cny"]))
    exposure_name = "d6_factor_exposures.parquet" if window == "design" else "d9_v2_factor_exposures.parquet"
    saved_exposures = read_frame(access, reports / exposure_name, "trade_date", start, end)
    industries = sorted(c for c in saved_exposures if c.startswith("industry_") and c != "industry_factor" and c != "industry_name")
    styles = [f"style_{x}" for x in config["d6"]["style_factors"]]
    if len(industries) != 31 or len(styles) != 6:
        raise DiagnosticError("D10 requires frozen 31 industries and 6 styles")
    if window == "design":
        exposures = saved_exposures
    else:
        print("D10 oos: rebuild DAILY D6 exposures, check equality at frozen weekly dates", flush=True)
        benchmark = read_frame(access, Path(config["data"]["tushare_raw_dir"]) / "index_daily_H00905_CSI.parquet",
                               "trade_date", history_start, end).sort_values("trade_date")
        unique_frame(benchmark, ["trade_date"], "benchmark")
        benchmark["benchmark_return"] = benchmark.close.pct_change(fill_method=None)
        basic = read_years(access, config, "daily_basic", history_start, end)
        if "instrument" not in basic:
            basic["instrument"] = basic.ts_code.map(tushare_to_qlib)
        basic = basic.loc[basic.instrument.isin(instruments)]
        d6 = config["d6"]
        raw = compute_raw_style_history(market[["trade_date", "instrument", "close"]], benchmark,
            basic, beta_window=int(d6["beta_window_trade_days"]),
            beta_minimum_observations=int(d6["beta_minimum_observations"]),
            momentum_window=int(d6["momentum_window_trade_days"]), momentum_skip=int(d6["momentum_skip_trade_days"]),
            liquidity_window=int(d6["liquidity_window_trade_days"]),
            liquidity_minimum_observations=int(d6["liquidity_minimum_observations"]))
        mapping = pd.read_parquet(access.root / d6["industry_mapping_file"])
        catalog = pd.read_parquet(access.root / d6["industry_catalog_file"])
        exposures, imputation, _, _ = assemble_exposures(weights, raw, mapping, catalog,
            pd.DatetimeIndex(weights.trade_date.unique()), winsor_mad_multiple=float(d6["winsor_mad_multiple"]))
        keys = ["trade_date", "instrument"]
        observed = exposures.loc[exposures.trade_date.isin(saved_exposures.trade_date.unique())].set_index(keys).sort_index()
        frozen = saved_exposures.set_index(keys).sort_index()
        if not observed.index.equals(frozen.index):
            raise DiagnosticError("daily/weekly exposure keys disagree")
        error = require_close(observed[industries + styles + ["market_cap", "benchmark_weight"]],
            frozen[industries + styles + ["market_cap", "benchmark_weight"]], "frozen weekly exposure parity", atol=1e-8, rtol=1e-10)
        output.json("oos_exposure_parity.json", {"weekly_rows": len(frozen), "maximum_error": error,
                                               "passed": True, "daily_dates": exposures.trade_date.nunique()})
        output.parquet("oos_daily_exposures.parquet", exposures)
        output.parquet("oos_daily_style_imputation.parquet", imputation)
    targets = {i: read_frame(access, reports / f"{prefix}_layer_{i}_targets.parquet", "signal_date", start, end) for i in range(7)}
    final_name = "d7_target_weights.parquet" if window == "design" else "d9_v2_target_weights.parquet"
    final_target = read_frame(access, reports / final_name, "signal_date", start, end)
    keys = ["signal_date", "instrument"]
    a, b = targets[6].set_index(keys).sort_index(), final_target.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise DiagnosticError("layer6/final target keys differ")
    require_close(a.target_weight, b.target_weight, "layer6/final target parity", atol=1e-10, rtol=0)
    for layer in [1, 2]:
        other = targets[layer].set_index(keys).sort_index()
        top = targets[0].set_index(keys).sort_index()
        if not other.index.equals(top.index):
            raise DiagnosticError("L0-L2 target key mismatch")
        require_close(other.target_weight, top.target_weight, "L0-L2 identical targets", atol=0, rtol=0)
    signals = pd.DatetimeIndex(final_target.signal_date.unique()).sort_values()
    calendar = pd.DatetimeIndex(strategy.trade_date.unique()).sort_values()
    executions = {}
    for signal in signals:
        later = calendar[calendar > signal]
        if not len(later):
            raise DiagnosticError("missing frozen T+1 execution date")
        executions[signal] = later[0]
    by_signal = pd.Series(executions)
    for layer, target in targets.items():
        unique_frame(target, keys, "targets")
        if set(target.signal_date) != set(signals):
            raise DiagnosticError("layer has missing signal dates")
        if "execution_date" in target:
            if not target.execution_date.equals(target.signal_date.map(by_signal)):
                raise DiagnosticError(f"L{layer}: frozen execution schedule mismatch")
    price_matrix = build_execution_price_matrix(market)
    refs = {}
    for layer in [0, 6]:
        refs[layer] = build_frozen_d7_pretrade_path(targets[layer], exposures, signals, by_signal,
            price_matrix, tolerance=float(config["d7"]["constraint_tolerance"]))
    prediction_name = "d5_walk_forward_predictions.parquet" if window == "design" else "d9_v2_oos_predictions.parquet"
    predictions = read_frame(access, reports / prediction_name, "sample_date", start, end)
    sample_audit = audit_prediction_universe(predictions, weights, market, calendar, int(config["label"]["horizon_trade_days"]))
    output.parquet(f"{window}_prediction_universe_audit.parquet", sample_audit)
    audit_name = "d8_target_audit.parquet" if window == "design" else "d9_v2_ladder_target_audit.parquet"
    target_audit = read_frame(access, reports / audit_name, "signal_date", start, end)
    return dict(start=start, end=end, prefix=prefix, weights=weights, flags=flags, market=market,
        strategy=strategy, ideal=ideal, exposures=exposures, industries=industries, styles=styles,
        targets=targets, references=refs, executions=by_signal, target_audit=target_audit, calendar=calendar,
        official=official)


def audit_prediction_universe(predictions, weights, market, calendar, horizon):
    """Quantify endpoint availability selection without computing new scores."""
    unique_frame(predictions, ["sample_date", "instrument"], "saved predictions")
    windows = make_label_windows(predictions.sample_date.unique(), calendar, horizon).set_index("sample_date")
    vwap = market.pivot(index="trade_date", columns="instrument", values="vwap")
    members = {d: g.set_index("instrument").weight for d, g in weights.groupby("trade_date")}
    rows = []
    for date, group in predictions.groupby("sample_date"):
        if date not in windows.index:
            raise DiagnosticError("saved prediction lacks its original forward label window")
        current = members[date]
        if not group.instrument.isin(current.index).all():
            raise DiagnosticError("saved prediction contains nonmember stocks")
        interval = windows.loc[date]
        entry = vwap.loc[interval.label_start_timestamp].reindex(current.index)
        exit_price = vwap.loc[interval.label_end_timestamp].reindex(current.index)
        available = entry.gt(0) & exit_price.gt(0) & np.isfinite(entry) & np.isfinite(exit_price)
        present = current.index.isin(group.instrument)
        rows.append({"sample_date": date, "label_start_timestamp": interval.label_start_timestamp,
            "label_end_timestamp": interval.label_end_timestamp, "members": len(current),
            "predicted": int(present.sum()), "missing_predictions": int((~present).sum()),
            "future_endpoint_unavailable": int((~available).sum()),
            "missing_prediction_and_future_endpoint_unavailable": int((~present & ~available).sum()),
            "missing_prediction_despite_available_endpoints": int((~present & available).sum()),
            "predicted_despite_missing_endpoint": int((present & ~available).sum()),
            "missing_prediction_benchmark_weight": float(current[~present].sum()),
            "finding": "inherited label-before-feature join; association counts, no revised predictions"})
    return pd.DataFrame(rows)


def capital_slice(frame, capital, label):
    if "capital" not in frame:
        raise DiagnosticError(f"{label}: capital column missing")
    selected = frame.loc[frame.capital.eq(capital)].copy()
    if selected.empty:
        raise DiagnosticError(f"{label}: missing capital scenario {capital}")
    return selected


def diagnose_layer(window, layer, capital, inputs, daily, trades, path, previous_path):
    targets = inputs["targets"]
    groups = {pd.Timestamp(d): g.set_index("instrument").target_weight for d, g in targets[layer].groupby("signal_date")}
    prior_targets = None if layer == 0 else {pd.Timestamp(d): g.set_index("instrument").target_weight for d, g in targets[layer - 1].groupby("signal_date")}
    members = {pd.Timestamp(d): g.instrument for d, g in inputs["weights"].groupby("trade_date")}
    trade_groups = {pd.Timestamp(d): g for d, g in trades.groupby("signal_date")}
    account = path.daily.set_index("trade_date")
    records = []
    frozen_audit = inputs["target_audit"].loc[inputs["target_audit"].layer.eq(layer)].set_index("signal_date")
    for signal, target in groups.items():
        date = pd.Timestamp(inputs["executions"].loc[signal])
        frozen = inputs["references"][6][signal]
        reference = inputs["references"][6 if layer >= 3 else 0][signal]
        split = turnover_split(target, reference, members[signal])
        # Saved audit names are part of the frozen D8 contract.
        require_close([split["target_total_turnover"]], [frozen_audit.loc[signal, "target_single_sided_turnover"]],
                      "target turnover reconstruction", atol=2e-7)
        orders = trade_groups.get(signal)
        if orders is None:
            raise DiagnosticError("missing signal's order ledger")
        execution_split = executed_turnover_split(orders, members[signal], float(account.loc[date, "pretrade_nav"]))
        live = path.pretrade[signal]
        current_eod = path.ending_weights.loc[date].copy()
        current_eod.loc["CASH"] = account.loc[date, "cash_weight"]
        row = {"window": window, "layer": layer, "capital": capital, "signal_date": signal,
            "execution_date": date, "target_reference": "frozen_final_target_drift" if layer >= 3 else "own_ideal_target_drift",
            **split, **execution_split,
            "target_distance_to_frozen_w0": half_l1(target, frozen),
            "live_pretrade_distance_to_frozen_w0": half_l1(live, frozen),
            "target_distance_to_previous_layer": half_l1(target, prior_targets[signal]) if prior_targets is not None else np.nan,
            "live_pretrade_distance_to_previous_layer": half_l1(live, previous_path.pretrade[signal]) if previous_path is not None else np.nan,
            "eod_distance_to_frozen_w0": half_l1(current_eod, frozen),
            "executed_cost_cny": float(orders.total_cost.sum()),
            "forced_exit_cost_cny": float(orders.loc[~orders.instrument.isin(members[signal]) & orders.executed_trade_amount.lt(0), "total_cost"].sum()),
            "blocked_orders": int(orders.blocked_by_tradability.sum()),
            "blocked_desired_notional_cny": float(orders.loc[orders.blocked_by_tradability, "desired_trade_amount"].abs().sum()),
            "participation_limited_orders": int(orders.participation_limited.sum()),
            "cash_limited_orders": int(orders.cash_limited.sum())}
        if previous_path is not None:
            previous_eod = previous_path.ending_weights.loc[date].copy()
            previous_eod.loc["CASH"] = previous_path.daily.set_index("trade_date").loc[date, "cash_weight"]
            row["eod_distance_to_previous_layer"] = half_l1(current_eod, previous_eod)
        records.append(row)
    detail = pd.DataFrame(records)
    metrics = {"window": window, "layer": layer, "capital": capital, **path_gross_net(daily, capital),
        "target_reference": detail.target_reference.iloc[0], "signals": len(detail),
        "maximum_nav_error_cny": float(path.daily.nav_error_cny.max()),
        "maximum_accounting_residual": float(path.daily.accounting_residual.abs().max())}
    for column in detail.select_dtypes(include="number").columns:
        if column in {"layer", "capital"} or not detail[column].notna().any():
            continue
        metrics[column + "_mean"] = float(detail[column].mean())
        metrics[column + "_median"] = float(detail[column].median())
        metrics[column + "_max"] = float(detail[column].max())
    metrics["target_live_gap_is_not_pure_blockage"] = True
    return metrics, detail


def layer2_comparison(window, capital, daily1, daily2, path1, path2, market, output):
    if not daily1.trade_date.equals(daily2.trade_date):
        raise DiagnosticError("L1-L2 date mismatch; do not align by silently dropping dates")
    require_close(daily1.benchmark_return, daily2.benchmark_return, "L1-L2 benchmark parity")
    a, b = path1.daily.set_index("trade_date"), path2.daily.set_index("trade_date")
    comparison = daily2[["trade_date", "portfolio_return"]].copy()
    comparison["benchmark_return"] = daily1.portfolio_return.to_numpy()  # comparison portfolio is L1
    comparison["holding_path_delta"] = (b.holding_return - a.holding_return).to_numpy()
    comparison["execution_pnl_delta"] = (b.execution_return - a.execution_return).to_numpy()
    comparison["cost_saving_delta"] = (a.cost_return - b.cost_return).to_numpy()
    columns = ["holding_path_delta", "execution_pnl_delta", "cost_saving_delta"]
    comparison["closure_residual"] = comparison.portfolio_return - comparison.benchmark_return - comparison[columns].sum(axis=1)
    require_close(comparison.closure_residual, np.zeros(len(comparison)), "L2-L1 paired daily PnL closure")
    monthly = linked_monthly(comparison, columns + ["closure_residual"])
    tag = f"{window}_{int(capital)}_l2_audit"
    output.parquet(tag + "_daily.parquet", comparison)
    output.parquet(tag + "_monthly.parquet", monthly)
    # Rank stock-level gross path contributions; these are associations, not causal treatment effects.
    mark, execution = price_panels(market)
    cc = mark / mark.shift(1) - 1
    intraday = mark / execution - 1
    holding = (path2.beginning_weights - path1.beginning_weights) * cc.reindex_like(path2.beginning_weights)
    trading = (path2.trade_weights - path1.trade_weights) * intraday.reindex_like(path2.trade_weights)
    stock = pd.DataFrame({"instrument": holding.columns,
        "sum_daily_holding_path_delta": holding.sum().to_numpy(),
        "sum_daily_execution_pnl_delta": trading.sum().to_numpy()})
    blocked = path2.orders.loc[path2.orders.blocked_by_tradability].groupby("instrument").size()
    stock["l2_blocked_order_count"] = stock.instrument.map(blocked).fillna(0).astype(int)
    stock["sum_daily_gross_path_delta"] = stock.sum_daily_holding_path_delta + stock.sum_daily_execution_pnl_delta
    output.parquet(tag + "_stock_associations.parquet", stock.sort_values("sum_daily_gross_path_delta", ascending=False))
    # Explicitly show which existing states were observed at signal vs execution.
    flags = market.set_index(["trade_date", "instrument"])
    state_rows = []
    for order in path2.orders.itertuples(index=False):
        key = (order.signal_date, order.instrument)
        signal_buy = bool(flags.loc[key, "can_buy"]) if key in flags.index else None
        signal_sell = bool(flags.loc[key, "can_sell"]) if key in flags.index else None
        current = flags.loc[(order.execution_date, order.instrument)]
        state_rows.append({"signal_date": order.signal_date, "execution_date": order.execution_date,
            "instrument": order.instrument, "signal_can_buy": signal_buy, "signal_can_sell": signal_sell,
            "execution_can_buy": bool(current.can_buy), "execution_can_sell": bool(current.can_sell),
            "blocked_by_tradability": order.blocked_by_tradability,
            "state_source": "execution-day realized filter; not a signal-time eligibility guarantee"})
    output.parquet(tag + "_state_timing.parquet", pd.DataFrame(state_rows))
    return {"window": window, "capital": capital,
        "l2_minus_l1_monthly_annualized": float(monthly.excess_return.mean() * 12),
        **{c: float(monthly[c].mean() * 12) for c in columns},
        "maximum_closure_residual": float(comparison.closure_residual.abs().max()),
        "cost_formula_parity_checked": True, "blocked_orders_create_no_positions_or_fees": True,
        "interpretation": "exact paired realized-path accounting; no unique causal effect of the filter",
        "future_label_sample_selection": "confirmed inherited code limitation; see d10_errata.md"}


def plot_attribution(monthly, output, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 5))
    columns = ["industry", "style", "specific_net"]
    colors = ["#3978a8", "#df9d34", "#589468"]
    positive = np.zeros(len(monthly)); negative = positive.copy()
    x = np.arange(len(monthly))
    for col, color in zip(columns, colors):
        values = monthly[col].to_numpy() * 100
        ax.bar(x, values, bottom=np.where(values >= 0, positive, negative), label=col, color=color)
        positive += np.maximum(values, 0); negative += np.minimum(values, 0)
    ax.plot(x, monthly.excess_return * 100, color="black", linewidth=1, label="actual net excess")
    ax.plot(x, monthly.three_block_residual * 100, color="#a64754", linewidth=.8, label="index + unmodelled bridge")
    ticks = x[::max(1, len(x) // 16)]
    ax.set_xticks(ticks, monthly.month.iloc[ticks], rotation=45, ha="right")
    ax.set_ylabel("Monthly contribution (percentage points)")
    ax.set_title("D10 ex-post WLS attribution | D9-V2 decision remains No-Go")
    ax.axhline(0, color="gray", linewidth=.5); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(output.target(name), dpi=160); plt.close(fig)


def run_stresses(access, config, inputs, main, output):
    reports = Path(config["data"]["reports_dir"])
    start, end = inputs["start"], inputs["end"]
    capital = float(config["d9"]["main_capital_cny"])
    base_model = CostModel.from_config(config)
    rows = []
    impact = read_frame(access, reports / "d9_v2_impact2x_daily.parquet", "trade_date", start, end)
    impact_trades = read_frame(access, reports / "d9_v2_impact2x_trades.parquet", "execution_date", start, end)
    if not impact.trade_date.equals(main.trade_date):
        raise DiagnosticError("existing impact2x date range differs from main")
    impact_path = reconstruct_executed_path(impact, impact_trades, inputs["strategy"], capital,
                                          replace(base_model, impact_coefficient=2 * base_model.impact_coefficient))
    for scenario in ["impact2x_reused", "delay_one_day", "biweekly"]:
        if scenario == "impact2x_reused":
            daily, trades, path = impact, impact_trades, impact_path
            source = "existing reports/d9_v2_impact2x_*; no rerun"
        else:
            dispatched, schedule = stress_schedule(inputs["targets"][6], inputs["calendar"], scenario)
            result = run_vwap_backtest(inputs["strategy"], dispatched[["signal_date", "instrument", "target_weight"]], capital, base_model)
            daily = align_stress_daily(result.nav, main, capital)
            trades = result.trades.copy()
            # Keep the dispatch signal used by D3 and add the original source explicitly.
            map_source = schedule.dropna(subset=["dispatch_date"]).set_index("dispatch_date").source_signal_date
            trades["source_signal_date"] = trades.signal_date.map(map_source)
            expected = schedule.dropna(subset=["dispatch_date"]).set_index("dispatch_date").execution_date
            if not trades.execution_date.equals(trades.signal_date.map(expected)):
                raise DiagnosticError("stress actual execution dates violate fixed schedule")
            path = reconstruct_executed_path(daily, trades, inputs["strategy"], capital, base_model)
            output.parquet(f"stress_{scenario}_schedule.parquet", schedule)
            output.parquet(f"stress_{scenario}_targets.parquet", dispatched)
            source = "frozen-target execution schedule sensitivity; no alpha refit or reoptimization"
        metrics, monthly = summarize_monthly_performance(daily, newey_west_lags=[3, 4, 5, 6])
        # Quantify membership retention caused by the changed execution schedule.
        membership = inputs["weights"].pivot(index="trade_date", columns="instrument", values="weight").notna()
        outside = path.ending_weights.where(~membership.reindex_like(path.ending_weights).fillna(False), 0).sum(axis=1)
        outside_rows = pd.DataFrame({"trade_date": outside.index, "outside_current_membership_weight": outside.to_numpy()})
        output.parquet(f"stress_{scenario}_membership.parquet", outside_rows)
        output.parquet(f"stress_{scenario}_daily.parquet", daily)
        output.parquet(f"stress_{scenario}_trades.parquet", trades)
        output.parquet(f"stress_{scenario}_monthly.parquet", monthly)
        output.parquet(f"stress_{scenario}_accounting.parquet", path.daily)
        rows.append({"scenario": scenario, "source": source, **metrics,
                     "maximum_outside_membership_weight": float(outside.max()),
                     "decision": "No-Go", "decision_changed": False})
    return rows


def run_diagnostics(access, config, output):
    summaries, ladder_details, layer2, attribution = [], [], [], {}
    reports = Path(config["data"]["reports_dir"])
    base_model = CostModel.from_config(config)
    main_capital = float(config["d9"]["main_capital_cny"])
    frozen_summary = json.loads((access.root / reports / "d9_v2_summary.json").read_text())
    main_saved = read_frame(access, reports / "d9_v2_main_daily.parquet", "trade_date",
                            config["windows"]["oos_start"], config["windows"]["oos_end"])
    main_monthly = read_frame(access, reports / "d9_v2_main_monthly.parquet", "month_end",
                              config["windows"]["oos_start"], config["windows"]["oos_end"])
    observed_metrics, observed_monthly = summarize_monthly_performance(main_saved, newey_west_lags=[3, 4, 5, 6])
    require_close(observed_metrics["annualized_excess_return"], frozen_summary["performance"]["annualized_excess_return"], "D9 main monthly annual excess")
    require_close(observed_metrics["information_ratio"], frozen_summary["performance"]["information_ratio"], "D9 main IR")
    if not observed_monthly.month.equals(main_monthly.month):
        raise DiagnosticError("D9 saved monthly keys differ")
    require_close(observed_monthly[["portfolio_return", "benchmark_return", "excess_return"]],
                  main_monthly[["portfolio_return", "benchmark_return", "excess_return"]], "D9 main monthly parity")
    stresses = []
    for window in ["design", "oos"]:
        inputs = load_window(access, config, window, output)
        expected_layers = json.loads((access.root / reports / ("d8_summary.json" if window == "design" else "d9_v2_summary.json")).read_text())
        expected_layers = expected_layers["layers" if window == "design" else "oos_ladder"]
        for capital in map(float, config["portfolio"]["capital_scenarios"]):
            previous_path = previous_daily = None
            for layer in range(7):
                print(f"D10 {window} / {capital:,.0f} / L{layer}: reconcile ledger and diagnose", flush=True)
                prefix = f"{inputs['prefix']}_layer_{layer}"
                daily = capital_slice(read_frame(access, reports / f"{prefix}_daily.parquet", "trade_date", inputs["start"], inputs["end"]), capital, "daily").reset_index(drop=True)
                trades = capital_slice(read_frame(access, reports / f"{prefix}_trades.parquet", "execution_date", inputs["start"], inputs["end"]), capital, "trades")
                official_returns = inputs["official"].set_index("trade_date").benchmark_return.reindex(daily.trade_date).copy()
                official_returns.iloc[0] = 0.0  # frozen D8 first-invested-day convention
                require_close(daily.benchmark_return, official_returns, "saved benchmark / official H00905 parity")
                selected_market = inputs["ideal"] if layer < 2 else inputs["strategy"]
                model = CostModel.zero_cost(float(config["d8"]["ideal_participation_limit"])) if layer == 0 else replace(base_model, participation_limit=float(config["d8"]["ideal_participation_limit"])) if layer == 1 else base_model
                path = reconstruct_executed_path(daily, trades, selected_market, capital, model)
                metrics, detail = diagnose_layer(window, layer, capital, inputs, daily, trades, path, previous_path)
                if capital == main_capital:
                    require_close([metrics["net_excess_cagr_difference"]], [expected_layers[layer]["annualized_excess_return"]], "frozen ladder CAGR")
                summaries.append(metrics); ladder_details.append(detail)
                tag = f"{window}_{int(capital)}_layer_{layer}"
                output.parquet(tag + "_accounting.parquet", path.daily)
                output.parquet(tag + "_order_audit.parquet", path.orders)
                if layer == 2:
                    layer2.append(layer2_comparison(window, capital, previous_daily, daily,
                                                    previous_path, path, inputs["strategy"], output))
                if layer == 6 and capital == main_capital:
                    if window == "oos":
                        if not daily.trade_date.equals(main_saved.trade_date):
                            raise DiagnosticError("L6/main daily dates differ")
                        require_close(daily[["nav", "cash", "portfolio_return", "benchmark_return"]],
                                      main_saved[["nav", "cash", "portfolio_return", "benchmark_return"]], "L6/main daily parity")
                    print(f"D10 {window}: daily ex-post WLS attribution", flush=True)
                    d, m, f, audit, result = attribute_executed_path(daily, path, inputs["strategy"], inputs["exposures"], inputs["industries"], inputs["styles"])
                    attribution[window] = result
                    for suffix, frame in [("daily", d), ("monthly", m), ("factor_returns", f), ("regression_audit", audit)]:
                        output.parquet(f"{window}_attribution_{suffix}.parquet", frame)
                    plot_attribution(m, output, f"{window}_attribution_monthly.png")
                previous_path, previous_daily = path, daily
            del previous_path, previous_daily, path
        if window == "oos":
            print("D10 oos: versioned execution sensitivity tests", flush=True)
            stresses = run_stresses(access, config, inputs, main_saved, output)
    ladder = pd.DataFrame(summaries)
    output.parquet("ladder_decomposition.parquet", ladder)
    output.parquet("ladder_by_signal.parquet", pd.concat(ladder_details, ignore_index=True))
    output.json("layer2_audit.json", {"comparisons": layer2})
    output.json("stress_summary.json", {"scenarios": stresses, "decision": "No-Go"})
    july = observed_monthly.loc[observed_monthly.month.eq("2026-07")]
    if len(july) != 1:
        raise DiagnosticError("2026 July monthly observation missing")
    july_values = july[["month", "portfolio_return", "benchmark_return", "excess_return"]].iloc[0].to_dict()
    output.json("july_2026.json", july_values)
    passed = bool(len(ladder) == 28 and all(x["three_block_residual_passed"] for x in attribution.values()))
    summary = {"version": "D10-v1", "purpose": "post-No-Go diagnosis only", "decision": "No-Go",
        "decision_changed": False, "main_model_refitted": False, "main_targets_reoptimized": False,
        "d9_v2_classification": "post_oos_methodology_amendment", "frozen_d9_v2_performance": frozen_summary["performance"],
        "ladder_rows": len(ladder), "ladder2_comparisons": layer2, "attribution": attribution,
        "stress_scenarios": stresses, "july_2026": july_values,
        "diagnostic_acceptance": {"all_28_layer_capital_windows": len(ladder) == 28,
            "accounting_reconciled": True, "stresses_versioned": len(stresses) == 3,
            "three_block_residual_under_0_5pct_per_year": all(x["three_block_residual_passed"] for x in attribution.values()),
            "passed_before_final_input_integrity_check": passed},
        "limitations": ["Fixed SW2021 is non-PIT, especially approximate before 2021.",
            "Net specific = gross stock residual minus explicitly disclosed costs; no index bridge is assigned to specific alpha.",
            "Targets L3-L6 share frozen w0; live paths differ. Target minus executed turnover is not purely blocked execution.",
            "Executed mandatory half-sell leg and target forced-exit-plus-replacement are distinct conventions.",
            "Inherited future-label availability selection and execution-day eligibility are registered in d10_errata.md.",
            "Delay/biweekly reuse fixed targets, do not certify feasibility of a freshly optimized strategy."]}
    return summary, ladder


def write_report(summary, ladder, output):
    lines = ["# D10 归因与诊断", "", "**D9-V2：No-Go，结论不变。以下为事后诊断。**", "",
             "D9-V2 月度年化超额 3.1081%；阶梯 CAGR 差额 3.6190%，两个口径分别保留。", "",
             "## 阶梯分解（主资金 1000 万）", "",
             "费前收益为实际持仓路径的费用加回；成本拖累是该路径费前/费后年化差。", "",
             "| 窗口 | 层 | 净超额 | 费前超额 | 成本拖累 | 目标总换手均值 | 实际换手均值 | 目标强制换手均值 | 目标主动换手均值 | 目标距 w0 均值 | 实际事前距 w0 均值 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ladder.loc[ladder.capital.eq(10_000_000)].itertuples(index=False):
        lines.append(f"| {r.window} | {r.layer} | {r.net_excess_cagr_difference:.2%} | {r.gross_excess_cagr_difference:.2%} | {r.cost_drag_cagr_difference:.2%} | {r.target_total_turnover_mean:.2%} | {r.executed_total_turnover_mean:.2%} | {r.target_mandatory_turnover_mean:.2%} | {r.target_discretionary_turnover_mean:.2%} | {r.target_distance_to_frozen_w0_mean:.2%} | {r.live_pretrade_distance_to_frozen_w0_mean:.2%} |")
    lines += ["", "2000 万资金与相邻层的目标/实际持仓距离详见 ladder_decomposition.parquet 和逐信号明细。",
              "实际强制退出按半个卖出腿列示，其余买卖腿单列；不得将这两个数字与 V2 满仓强制退出及替代买入的目标定义混用。", "",
              "## 第二层反常变化", "",
              "| 窗口/资金 | L2−L1 月度年化差 | 原持仓路径贡献 | 成交后日内贡献 | 成本节约贡献 |", "|---|---:|---:|---:|---:|"]
    for r in summary["ladder2_comparisons"]:
        lines.append(f"| {r['window']}/{r['capital']:,.0f} | {r['l2_minus_l1_monthly_annualized']:.2%} | {r['holding_path_delta']:.2%} | {r['execution_pnl_delta']:.2%} | {r['cost_saving_delta']:.2%} |")
    lines += ["", "这是两个完整实际路径的精确记账分解，无法单独证明过滤器的因果效果。逐笔费用、状态时间、阻单和股票路径关联已独立落盘。", "", "## 归因", ""]
    for window, r in summary["attribution"].items():
        share = r["specific_share_net"]
        lines.append(f"- {window}：费后特异性占比 {'未定义（超额非正）' if share is None else format(share, '.2%')}；三块残差年化平均绝对值 {r['annualized_mean_absolute_three_block_residual']:.3%}；最差日历年 {r['maximum_calendar_year_annualized_absolute_residual']:.3%}；残差验收 {r['three_block_residual_passed']}。")
        lines += ["", f"![{window} monthly attribution]({window}_attribution_monthly.png)", ""]
    lines += ["特异性净贡献为 WLS 个股残差减去明确披露的真实交易费用；费用前残差也单列。官方指数桥接、未覆盖股票/行业和陈旧暴露的贡献独立披露，不塞进特异性 alpha。", "", "## 压力测试", "",
              "| 情景 | 月度年化超额 | IR | 最大相对回撤 |", "|---|---:|---:|---:|"]
    for r in summary["stress_scenarios"]:
        lines.append(f"| {r['scenario']} | {r['annualized_excess_return']:.2%} | {r['information_ratio']:.3f} | {r['maximum_relative_drawdown']:.2%} |")
    j = summary["july_2026"]
    lines += ["", "冲击翻倍复用 V2，延迟与双周均为冻结目标的执行排期敏感性，保留现金等待日和主结果全部比较日期；调样滞留权重单列。", "",
        "## 2026 年 7 月", "",
        f"模型绝对收益 **{j['portfolio_return']:.2%}**；中证500全收益指数 **{j['benchmark_return']:.2%}**；当月超额 **{j['excess_return']:.2%}**。", "",
        "## 验收与勘误", "", f"诊断验收：{summary['diagnostic_acceptance']['passed']}。该值只表示 D10 诊断完整性，不能改变研究 No-Go。",
        "输入前后 SHA256 校验及 frozen-oos 核验详见 completion.json。既有 D1–D9 文件保留；静态发现与口径澄清详见仓库 D10勘误登记.md。", ""]
    output.text("d10_report.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="D10 read-only post-No-Go diagnostics")
    parser.add_argument("--preflight", action="store_true", help="check branch, old hashes, inputs and dependencies; no Qlib return read")
    parser.add_argument("--run-id", default="d10_v1", help="new exclusive reports/d10/<id>; never overwrites")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    baseline = json.loads((ROOT / "config/d10_baseline.json").read_text())
    checked = preflight(ROOT, config, baseline)
    if args.preflight:
        print(json.dumps(checked, ensure_ascii=False, indent=2)); return
    output = D10Output(ROOT, args.run_id)
    output.json("started.json", {**checked, "run_id": args.run_id, "classification": "D10 ex-post diagnostic only"})
    print("D10: hashing protected inputs and Qlib bins; originals are read only", flush=True)
    before = snapshot_inputs(ROOT, config)
    output.json("inputs_before.json", before)
    summary = ladder = error = None
    try:
        summary, ladder = run_diagnostics(ResearchDataAccess(ROOT), config, output)
    except Exception as exc:
        error = exc
        output.json("failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "decision": "No-Go"})
        output.json("runtime_errata.json", {"status": "requires_review", "observed_discrepancy": str(exc),
            "historical_results_overwritten": False, "corrected_performance": None,
            "interpretation": "Input/code/accounting mismatch, not permission to repair or rerun D9.", "decision": "No-Go"})
    finally:
        after = snapshot_inputs(ROOT, config)
        output.json("inputs_after.json", after)
        try:
            assert_snapshot_unchanged(before, after)
            verify_repository(ROOT, baseline)
        except Exception as exc:
            output.json("integrity_failure.json", {"error": str(exc), "decision": "No-Go"})
            raise
    if error is not None:
        raise error
    passed = bool(summary["diagnostic_acceptance"].pop("passed_before_final_input_integrity_check"))
    summary["diagnostic_acceptance"].update({"inputs_unchanged": True, "passed": passed})
    output.json("d10_summary.json", summary)
    write_report(summary, ladder, output)
    output.json("completion.json", {"status": "completed" if passed else "diagnostic_acceptance_failed",
        "inputs_unchanged": True, "decision": "No-Go", "decision_changed": False})
    print(f"D10 report: {output.path / 'd10_report.md'}; diagnostic_passed={passed}; D9-V2 remains No-Go")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

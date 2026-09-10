"""Pure Day-5 comparisons for nominal diagnostics and dynamic impact stress."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from scripts.d11_diagnose import check_parity
from src.d10_core import DiagnosticError, unique_frame
from src.d12_core import ImpactEstimator
from src.d12_repo import D12Error


PERFORMANCE_METRICS = (
    "annualized_excess_return",
    "annualized_tracking_error",
    "information_ratio",
    "newey_west_t_minimum",
    "maximum_relative_drawdown",
)


def impact2x_configuration(config):
    """Double execution impact only; keep R1's registered decision estimator at 0.5."""
    if float(config["costs"]["impact"]["coefficient"]) != .5:
        raise D12Error("Day 5 requires nominal execution impact k=0.5")
    stressed = deepcopy(config)
    stressed["costs"]["impact"]["coefficient"] = 1.0
    decision = ImpactEstimator.from_config(config, k=.5)
    if decision.k != .5 or float(stressed["costs"]["impact"]["coefficient"]) != 1.:
        raise D12Error("Day-5 stress specification changed")
    return stressed, decision


def daily_from_simulation(result, benchmark_daily, initial_capital):
    daily = result.nav.copy()
    daily = daily.merge(
        benchmark_daily[["trade_date", "benchmark_return"]],
        on="trade_date", how="left", validate="one_to_one")
    daily["portfolio_return"] = daily.nav / daily.nav.shift(
        1, fill_value=float(initial_capital)) - 1
    if daily.empty or not np.isfinite(
            daily[["portfolio_return", "benchmark_return", "nav", "cash"]]).all().all():
        raise DiagnosticError("Day-5 simulated daily path is empty or non-finite")
    return daily


def nominal_replay_parity(result, daily, saved):
    """Require a fresh R1 nominal replay to reproduce all decision and account states."""
    if not daily.trade_date.equals(saved["daily"].trade_date):
        raise DiagnosticError("R1 nominal replay date mismatch")
    errors = {
        "daily": check_parity(daily, saved["daily"], ["trade_date"],
                              ["nav", "cash", "transaction_cost"],
                              "R1 nominal account", atol=1e-4),
        "returns": check_parity(daily, saved["daily"], ["trade_date"],
                                ["single_sided_turnover", "portfolio_return"],
                                "R1 nominal returns"),
        "targets": check_parity(result.targets, saved["targets"],
                                ["signal_date", "instrument"],
                                ["target_weight", "pretrade_weight"],
                                "R1 nominal targets"),
        "orders": check_parity(result.orders, saved["orders"],
                               ["signal_date", "instrument"],
                               ["order_units", "signal_nav"],
                               "R1 nominal orders", atol=1e-5),
        "order_impact": check_parity(
            result.orders, saved["orders"], ["signal_date", "instrument"],
            ["decision_coefficient", "expected_impact_fraction"],
            "R1 nominal order impact"),
        "trades": check_parity(result.trades, saved["trades"],
                               ["execution_date", "instrument"],
                               ["executed_units", "total_cost"],
                               "R1 nominal fills", atol=1e-5),
        "audits": check_parity(result.audits, saved["audits"], ["signal_date"],
                               ["target_active_turnover", "expected_impact_fraction",
                                "decision_impact_k"], "R1 nominal audits", atol=1e-8),
    }
    discrete = [
        (result.orders, saved["orders"], ["signal_date", "instrument"],
         ["execution_date", "signal_can_buy", "amount_buy_forbidden", "sigma_source"]),
        (result.trades, saved["trades"], ["execution_date", "instrument"],
         ["exchange_can_buy", "exchange_can_sell", "participation_limited"]),
        (result.audits, saved["audits"], ["signal_date"],
         ["solver_status", "fallback_used", "initial_construction_exempt"]),
    ]
    for new, old, keys, columns in discrete:
        left, right = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
        if not left.index.equals(right.index) or not left[columns].equals(right[columns]):
            raise DiagnosticError("R1 nominal replay changed a discrete decision or execution state")
    return {"passed": True, "maximum_errors": errors,
            "comparison": "fresh R1 k_decision=0.5/k_execution=0.5 replay vs saved r1_nominal_v1"}


def signal_calibration(predictions, labels, end, *, deciles=10):
    """Full-history evaluable deciles for the fixed R0/R1 prediction stream."""
    keys = ["sample_date", "instrument"]
    unique_frame(predictions, keys, "Day-5 predictions")
    unique_frame(labels, keys, "Day-5 labels")
    required = {"prediction", "expected_5d_excess", "feature_available"}
    if not required.issubset(predictions):
        raise DiagnosticError(f"Prediction calibration fields missing: {sorted(required-set(predictions))}")
    frame = predictions.merge(
        labels[keys + ["excess_return", "label_end_timestamp"]],
        on=keys, how="left", validate="one_to_one")
    frame = frame.loc[
        frame.label_end_timestamp.le(pd.Timestamp(end))
        & frame.feature_available.eq(True)
        & np.isfinite(frame.prediction)
        & np.isfinite(frame.expected_5d_excess)
        & np.isfinite(frame.excess_return)].copy()
    if frame.empty:
        raise DiagnosticError("No mature prediction labels for Day-5 calibration")

    def assign(series):
        ranks = series.rank(method="first").astype(int) - 1
        return np.floor(ranks * int(deciles) / len(series)).astype(int) + 1

    sizes = frame.groupby("sample_date").instrument.transform("size")
    frame = frame.loc[sizes.ge(int(deciles))].copy()
    frame["decile"] = frame.groupby("sample_date", group_keys=False)[
        "prediction"].apply(assign)
    table = frame.groupby("decile", as_index=False).agg(
        mean_expected_5d_excess=("expected_5d_excess", "mean"),
        mean_realized_5d_excess=("excess_return", "mean"),
        observations=("excess_return", "size"))
    if len(table) != int(deciles):
        raise DiagnosticError(f"Day-5 calibration produced {len(table)} deciles, expected {deciles}")
    table["calibration_error"] = (
        table.mean_realized_5d_excess - table.mean_expected_5d_excess)
    x = frame.expected_5d_excess.to_numpy(float)
    y = frame.excess_return.to_numpy(float)
    denominator = float(x @ x)
    scale = float(x @ y / denominator) if denominator > 0 else None
    means = table.mean_realized_5d_excess.to_numpy(float)
    summary = {
        "evaluable_rows": int(len(frame)),
        "dates": int(frame.sample_date.nunique()),
        "deciles": int(deciles),
        "top_minus_bottom_realized": float(means[-1] - means[0]),
        "adjacent_up_steps": int((np.diff(means) >= 0).sum()),
        "decile_spearman": float(table.decile.corr(
            table.mean_realized_5d_excess, method="spearman")),
        "zero_intercept_realized_on_expected_scale": scale,
        "decile_rmse": float(np.sqrt(np.mean(table.calibration_error.pow(2)))),
        "prediction_identity": "R0 and R1 reuse the same repaired-R0 predictions; IC/calibration are identical by design",
    }
    return table, summary


def performance_matrix(scenarios):
    rows = []
    for name, item in scenarios.items():
        performance = item["performance"]
        detail = item.get("performance_detail", {})
        execution = item.get("execution", {})
        row = {"scenario": name, "decision_k": item.get("decision_k"),
               "execution_k": item.get("execution_k")}
        row.update({metric: performance.get(metric) for metric in PERFORMANCE_METRICS})
        row["same_path_gross_annualized_excess"] = detail.get(
            "same_path_gross", {}).get("annualized_excess_return")
        row["cost_drag_annualized"] = detail.get("cost_drag_annualized")
        row["total_cost_cny"] = execution.get("total_cost_cny")
        row["impact_cost_cny"] = execution.get("costs_cny", {}).get("impact_cost")
        rows.append(row)
    return pd.DataFrame(rows)


def stress_effects(matrix):
    frame = matrix.set_index("scenario")
    pairs = [
        ("r1_nominal_minus_r0_nominal", "r1_nominal", "repaired_r0_nominal"),
        ("r0_impact2x_minus_nominal", "repaired_r0_impact2x", "repaired_r0_nominal"),
        ("r1_impact2x_minus_nominal", "r1_impact2x", "r1_nominal"),
        ("r1_minus_r0_under_impact2x", "r1_impact2x", "repaired_r0_impact2x"),
    ]
    rows = []
    for label, left, right in pairs:
        if left not in frame.index or right not in frame.index:
            raise D12Error(f"Stress matrix is missing {left!r} or {right!r}")
        row = {"comparison": label, "left": left, "right": right}
        for metric in PERFORMANCE_METRICS:
            row[metric + "_difference"] = float(frame.loc[left, metric] - frame.loc[right, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_comparison(attributions):
    rows = []
    for strategy, summary in attributions.items():
        annual = summary["annualized_contributions"]
        for component, value in annual.items():
            rows.append({"strategy": strategy, "measure": "annualized_" + component,
                         "value": value})
        for measure in ["specific_share_net", "specific_share_gross",
                        "annualized_mean_absolute_three_block_residual",
                        "maximum_calendar_year_annualized_absolute_residual",
                        "maximum_stale_exposure_l1"]:
            rows.append({"strategy": strategy, "measure": measure,
                         "value": summary.get(measure)})
        rows.append({"strategy": strategy, "measure": "three_block_residual_passed",
                     "value": bool(summary["three_block_residual_passed"])})
    return pd.DataFrame(rows)


def reference_gates(nominal, impact2x, attribution, config):
    thresholds = config["d9"]["thresholds"]
    rows = []
    definitions = [
        ("annualized_excess_return", "annualized_excess_return_min", ">="),
        ("information_ratio", "information_ratio_min", ">="),
        ("newey_west_t_minimum", "newey_west_t_min", ">="),
        ("maximum_relative_drawdown", "maximum_relative_drawdown_max", "<="),
    ]
    for strategy in ["repaired_r0", "r1"]:
        for metric, threshold_name, operator in definitions:
            value, threshold = nominal[strategy][metric], thresholds[threshold_name]
            passed = value >= threshold if operator == ">=" else value <= threshold
            rows.append({"strategy": strategy, "gate": metric, "value": value,
                         "operator": operator, "threshold": threshold, "passed": bool(passed)})
        share = attribution[strategy]["specific_share_net"]
        threshold = thresholds["specific_contribution_share_min"]
        rows.append({"strategy": strategy, "gate": "specific_share_net", "value": share,
                     "operator": ">=", "threshold": threshold,
                     "passed": bool(share is not None and share >= threshold)})
        stress = impact2x[strategy]["annualized_excess_return"]
        threshold = thresholds["doubled_impact_annualized_excess_min"]
        rows.append({"strategy": strategy, "gate": "impact2x_annualized_excess_return",
                     "value": stress, "operator": ">", "threshold": threshold,
                     "passed": bool(stress > threshold)})
    return pd.DataFrame(rows)

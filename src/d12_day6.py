"""Pure Day-6 evidence-review helpers; no model fitting or portfolio simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.d10_core import DiagnosticError, unique_frame
from src.d12_repo import D12Error


TARGET_YEARS = (2023, 2025)
DECILES = 10
FEATURE_REVIEW_THRESHOLDS = {
    "absolute_median_shift_iqr": 1.0,
    "missing_fraction_change": 0.05,
    "outside_reference_q01_q99_fraction": 0.10,
}


def validate_day6_freeze(experiments):
    """Refuse post-result changes to the fixed Day-6/R2/forward specification."""
    day6 = experiments.get("day6_freeze", {})
    r2 = experiments.get("r2", {})
    forward = day6.get("forward_recording", {})
    checks = {
        "day6_frozen": day6.get("frozen") is True,
        "target_years": tuple(day6.get("p2", {}).get("target_years", [])) == TARGET_YEARS,
        "deciles": int(day6.get("p2", {}).get("deciles", -1)) == DECILES,
        "causal_reference": day6.get("p2", {}).get("feature_reference")
            == "strictly earlier scored OOS years",
        "r2_frozen": r2.get("frozen") is True,
        "r2_not_run": r2.get("status")
            == "frozen_for_next_week_blocked_pending_start_conditions",
        "r2_ten_training_folds": r2.get("fold_resolution", {}).get("training_fold_years")
            == list(range(2017, 2027)),
        "r2_eight_scored_years": r2.get("fold_resolution", {}).get("scored_oos_years")
            == list(range(2019, 2027)),
        "forward_default_r0": forward.get("default_strategy") == "R0",
        "r1_independent_shadow": forward.get("r1_policy") == "independent shadow path",
        "no_return_without_fills": forward.get("no_fill_engine_policy")
            == "log scores and simulated orders; do not claim strategy return",
        "mature_labels_append_only": forward.get("mature_label_policy")
            == "append later; never overwrite the original score",
    }
    if not all(checks.values()):
        raise D12Error("Day-6 frozen specification changed: " + str(checks))
    return checks


def _assign_deciles(series, deciles=DECILES):
    ranks = series.rank(method="first").astype(int) - 1
    return np.floor(ranks * int(deciles) / len(series)).astype(int) + 1


def signal_year_diagnostics(predictions, labels, end, *, years=TARGET_YEARS,
                            deciles=DECILES):
    """Inspect fixed-score deciles/calibration in the two pre-registered P2 years."""
    years = tuple(int(year) for year in years)
    if years != TARGET_YEARS or int(deciles) != DECILES:
        raise DiagnosticError("Day-6 P2 years/deciles are frozen at 2023, 2025 and 10")
    keys = ["sample_date", "instrument"]
    unique_frame(predictions, keys, "Day-6 predictions")
    unique_frame(labels, keys, "Day-6 labels")
    required = {"prediction", "expected_5d_excess", "feature_available"}
    if not required.issubset(predictions):
        raise DiagnosticError("Day-6 prediction fields missing: "
                              + str(sorted(required - set(predictions))))
    frame = predictions.merge(
        labels[keys + ["excess_return", "label_end_timestamp"]], on=keys,
        how="left", validate="one_to_one")
    frame["sample_date"] = pd.to_datetime(frame.sample_date)
    frame["label_end_timestamp"] = pd.to_datetime(frame.label_end_timestamp)
    frame = frame.loc[
        frame.sample_date.dt.year.isin(years)
        & frame.label_end_timestamp.le(pd.Timestamp(end))
        & frame.feature_available.eq(True)
        & np.isfinite(frame.prediction)
        & np.isfinite(frame.expected_5d_excess)
        & np.isfinite(frame.excess_return)].copy()
    if set(frame.sample_date.dt.year.unique()) != set(years):
        raise DiagnosticError("Day-6 lacks mature evaluable rows for 2023 or 2025")
    sizes = frame.groupby("sample_date").instrument.transform("size")
    frame = frame.loc[sizes.ge(int(deciles))].copy()
    frame["decile"] = frame.groupby("sample_date", group_keys=False)[
        "prediction"].apply(lambda value: _assign_deciles(value, deciles))

    by_date = []
    for date, part in frame.groupby("sample_date", sort=True):
        means = part.groupby("decile").excess_return.mean()
        if len(means) != int(deciles):
            raise DiagnosticError(f"Day-6 {date.date()} did not produce ten deciles")
        by_date.append({
            "sample_date": date,
            "year": int(date.year),
            "rank_ic": float(part.prediction.corr(part.excess_return, method="spearman")),
            "top_minus_bottom_realized": float(means.loc[int(deciles)] - means.loc[1]),
            "observations": int(len(part)),
        })
    by_date = pd.DataFrame(by_date)
    frame["year"] = frame.sample_date.dt.year
    decile_table = frame.groupby(
        ["year", "decile"], as_index=False).agg(
            mean_prediction=("prediction", "mean"),
            mean_expected_5d_excess=("expected_5d_excess", "mean"),
            mean_realized_5d_excess=("excess_return", "mean"),
            observations=("excess_return", "size"),
            dates=("sample_date", "nunique"))
    decile_table["calibration_error"] = (
        decile_table.mean_realized_5d_excess - decile_table.mean_expected_5d_excess)

    rows = []
    for year in years:
        values = frame.loc[frame.sample_date.dt.year.eq(year)]
        dates = by_date.loc[by_date.year.eq(year)]
        decile = decile_table.loc[decile_table.year.eq(year)].sort_values("decile")
        x = values.expected_5d_excess.to_numpy(float)
        y = values.excess_return.to_numpy(float)
        denominator = float(x @ x)
        positive = dates.top_minus_bottom_realized.clip(lower=0)
        top_five = dates.nlargest(min(5, len(dates)), "top_minus_bottom_realized")
        rows.append({
            "year": year,
            "rows": int(len(values)),
            "dates": int(values.sample_date.nunique()),
            "rank_ic_mean": float(dates.rank_ic.mean()),
            "rank_ic_positive_fraction": float(dates.rank_ic.gt(0).mean()),
            "top_minus_bottom_mean": float(dates.top_minus_bottom_realized.mean()),
            "top_minus_bottom_median": float(dates.top_minus_bottom_realized.median()),
            "top_minus_bottom_positive_fraction": float(
                dates.top_minus_bottom_realized.gt(0).mean()),
            "best_five_positive_spread_share": (
                float(top_five.top_minus_bottom_realized.clip(lower=0).sum()
                      / positive.sum()) if positive.sum() > 0 else None),
            "best_date": dates.loc[dates.top_minus_bottom_realized.idxmax(),
                                   "sample_date"],
            "worst_date": dates.loc[dates.top_minus_bottom_realized.idxmin(),
                                    "sample_date"],
            "decile_spearman": float(decile.decile.corr(
                decile.mean_realized_5d_excess, method="spearman")),
            "adjacent_up_steps": int((np.diff(
                decile.mean_realized_5d_excess.to_numpy(float)) >= 0).sum()),
            "zero_intercept_realized_on_expected_scale": (
                float(x @ y / denominator) if denominator > 0 else None),
            "registered_issue": ("near_zero_signal_year" if year == 2023
                                 else "positive_rank_ic_negative_tail_spread"),
        })
    yearly = pd.DataFrame(rows)
    summary = {
        "target_years": list(years),
        "deciles": int(deciles),
        "observations": int(len(frame)),
        "dates": int(frame.sample_date.nunique()),
        "diagnostic_only": True,
        "model_or_cutpoints_changed": False,
        "prediction_identity": "repaired R0 and R1 reuse the same fixed predictions",
    }
    return decile_table, by_date, yearly, summary


def feature_distribution_review(features, feature_names, *, years=TARGET_YEARS,
                                thresholds=FEATURE_REVIEW_THRESHOLDS):
    """Compare each target year with strictly earlier scored OOS years.

    Flags are review screens, not automatic data-error or R2-start decisions.
    """
    years = tuple(int(year) for year in years)
    if years != TARGET_YEARS:
        raise DiagnosticError("Day-6 feature review years are frozen at 2023 and 2025")
    unique_frame(features, ["sample_date", "instrument"], "Day-6 feature matrix")
    frame = features.copy()
    frame["sample_date"] = pd.to_datetime(frame.sample_date)
    names = list(feature_names)
    if not names or not set(names).issubset(frame):
        raise DiagnosticError("Day-6 feature specification is empty or does not match the matrix")
    rows = []
    for year in years:
        target = frame.loc[frame.sample_date.dt.year.eq(year), names]
        reference = frame.loc[
            frame.sample_date.dt.year.lt(year) & frame.sample_date.dt.year.ge(2019), names]
        if target.empty or reference.empty:
            raise DiagnosticError(f"Day-6 feature review lacks target/reference rows for {year}")
        for name in names:
            target_values = pd.to_numeric(target[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
            reference_values = pd.to_numeric(reference[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
            target_finite = target_values.loc[np.isfinite(target_values)]
            reference_finite = reference_values.loc[np.isfinite(reference_values)]
            if reference_finite.empty:
                reference_q01 = reference_q25 = reference_median = np.nan
                reference_q75 = reference_q99 = np.nan
            else:
                reference_q01, reference_q25, reference_median, reference_q75, reference_q99 = (
                    reference_finite.quantile([.01, .25, .5, .75, .99]).to_numpy(float))
            target_median = float(target_finite.median()) if len(target_finite) else np.nan
            reference_iqr = reference_q75 - reference_q25
            median_difference = target_median - reference_median
            median_shift = (median_difference / reference_iqr
                            if np.isfinite(reference_iqr) and reference_iqr > 0 else np.nan)
            outside = (float(((target_finite < reference_q01)
                             | (target_finite > reference_q99)).mean())
                       if len(target_finite) and np.isfinite(reference_q01)
                       and np.isfinite(reference_q99) else np.nan)
            target_missing = float(target_values.isna().mean())
            reference_missing = float(reference_values.isna().mean())
            missing_change = target_missing - reference_missing
            reasons = []
            if np.isfinite(median_shift) and abs(median_shift) > float(
                    thresholds["absolute_median_shift_iqr"]):
                reasons.append("median_shift")
            if (np.isfinite(reference_iqr) and reference_iqr == 0
                    and np.isfinite(median_difference) and median_difference != 0):
                reasons.append("constant_reference_level_shift")
            if abs(missing_change) > float(thresholds["missing_fraction_change"]):
                reasons.append("missingness")
            if np.isfinite(outside) and outside > float(
                    thresholds["outside_reference_q01_q99_fraction"]):
                reasons.append("outside_reference_range")
            if not len(reference_finite) or not len(target_finite):
                reasons.append("insufficient_finite_values")
            rows.append({
                "year": year, "feature": name,
                "target_rows": int(len(target_values)),
                "reference_rows": int(len(reference_values)),
                "target_missing_fraction": target_missing,
                "reference_missing_fraction": reference_missing,
                "missing_fraction_change": missing_change,
                "target_median": target_median,
                "reference_median": reference_median,
                "reference_iqr": reference_iqr,
                "absolute_median_difference": abs(median_difference),
                "reference_q01": reference_q01,
                "reference_q25": reference_q25,
                "reference_q75": reference_q75,
                "reference_q99": reference_q99,
                "normalized_median_shift_iqr": median_shift,
                "target_outside_reference_q01_q99_fraction": outside,
                "review_flag": bool(reasons),
                "review_reasons": ";".join(reasons),
            })
    detail = pd.DataFrame(rows)
    summaries = []
    for year, part in detail.groupby("year", sort=True):
        shifts = part.normalized_median_shift_iqr.abs().replace([np.inf, -np.inf], np.nan)
        top = shifts.idxmax() if shifts.notna().any() else None
        summaries.append({
            "year": int(year), "features": int(len(part)),
            "review_flags": int(part.review_flag.sum()),
            "top_absolute_median_shift_feature": (part.loc[top, "feature"] if top is not None else None),
            "top_absolute_median_shift_iqr": (float(shifts.loc[top]) if top is not None else None),
            "maximum_missing_fraction_change_abs": float(part.missing_fraction_change.abs().max()),
            "maximum_outside_reference_fraction": float(
                part.target_outside_reference_q01_q99_fraction.max()),
            "interpretation": "descriptive review screen; not an automatic data-error finding",
        })
    return detail, pd.DataFrame(summaries)


def feature_extreme_observations(features, feature_detail, *, top_n=5):
    """Keep instrument/date evidence behind flagged feature-distribution screens."""
    if int(top_n) < 1:
        raise DiagnosticError("Day-6 feature extreme top_n must be positive")
    frame = features.copy()
    frame["sample_date"] = pd.to_datetime(frame.sample_date)
    rows = []
    flagged = feature_detail.loc[feature_detail.review_flag.eq(True)]
    for item in flagged.itertuples(index=False):
        part = frame.loc[frame.sample_date.dt.year.eq(int(item.year)),
                         ["sample_date", "instrument", item.feature]].copy()
        part["value"] = pd.to_numeric(part[item.feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        scale = float(item.reference_iqr) if np.isfinite(item.reference_iqr) else np.nan
        if np.isfinite(scale) and scale > 0:
            part["distance_from_reference_range_iqr"] = np.maximum.reduce([
                (float(item.reference_q01) - part.value) / scale,
                (part.value - float(item.reference_q99)) / scale,
                np.zeros(len(part)),
            ])
        else:
            part["distance_from_reference_range_iqr"] = (
                part.value - float(item.reference_median)).abs()
        part = part.nlargest(int(top_n), "distance_from_reference_range_iqr")
        for value in part.itertuples(index=False):
            rows.append({
                "year": int(item.year), "feature": item.feature,
                "sample_date": value.sample_date, "instrument": value.instrument,
                "value": value.value,
                "reference_q01": item.reference_q01,
                "reference_median": item.reference_median,
                "reference_q99": item.reference_q99,
                "distance_from_reference_range_iqr": value.distance_from_reference_range_iqr,
                "review_reasons": item.review_reasons,
                "interpretation": "extreme observation for manual review; not a data-error verdict",
            })
    return pd.DataFrame(rows, columns=[
        "year", "feature", "sample_date", "instrument", "value", "reference_q01",
        "reference_median", "reference_q99", "distance_from_reference_range_iqr",
        "review_reasons", "interpretation"])


def constraint_tradability_review(targets, audits, trades, config, strategy,
                                  *, years=TARGET_YEARS):
    """Summarize observed constraints/tradability without claiming counterfactual causality."""
    targets, audits, trades = targets.copy(), audits.copy(), trades.copy()
    for frame, column in [(targets, "signal_date"), (audits, "signal_date"),
                          (trades, "execution_date")]:
        frame[column] = pd.to_datetime(frame[column])
    limits = config["d7"]
    tolerance = float(limits["constraint_tolerance"]) * 5
    rows = []
    for year in years:
        target = targets.loc[targets.signal_date.dt.year.eq(year)]
        audit = audits.loc[audits.signal_date.dt.year.eq(year)]
        trade = trades.loc[trades.execution_date.dt.year.eq(year)]
        if target.empty or audit.empty:
            raise DiagnosticError(f"Day-6 {strategy} lacks target/audit rows in {year}")
        unfilled = (trade.unfilled_units.abs() * trade.signal_price
                    if {"unfilled_units", "signal_price"}.issubset(trade) else pd.Series(0., index=trade.index))
        rows.append({
            "strategy": strategy, "year": int(year),
            "decisions": int(audit.signal_date.nunique()),
            "target_rows": int(len(target)),
            "mean_buy_eligible_fraction": float(target.signal_can_buy.eq(True).mean()),
            "mean_prohibited_or_unknown_names": float(
                audit.prohibited_or_unknown_buy_names.mean()),
            "turnover_cap_binding_fraction": float(audit.target_active_turnover.ge(
                float(limits["single_sided_turnover_limit"]) - tolerance).mean()),
            "stock_limit_binding_fraction": float(audit.maximum_stock_active.ge(
                float(limits["stock_active_limit"]) - tolerance).mean()),
            "industry_limit_binding_fraction": float(audit.maximum_industry_active.ge(
                float(limits["industry_active_limit"]) - tolerance).mean()),
            "style_limit_binding_fraction": float(audit.maximum_style_active.ge(
                float(limits["style_active_limit"]) - tolerance).mean()),
            "mean_mandatory_turnover": float(audit.mandatory_turnover.mean()),
            "maximum_prohibited_buy_weight": float(audit.prohibited_buy_weight.max()),
            "trade_rows": int(len(trade)),
            "participation_limited_rows": int(trade.participation_limited.eq(True).sum()),
            "cash_scaled_buy_rows": int((
                trade.get("side", pd.Series("", index=trade.index)).eq("buy")
                & trade.cash_scale.lt(1 - 1e-12)
                & trade.executed_units.ne(0)).sum()),
            "unfilled_signal_cny": float(unfilled.sum()),
            "interpretation": "observed binding telemetry; not a constraint-relaxation counterfactual",
        })
    return pd.DataFrame(rows)


def monthly_concentration(monthly_three_way):
    """Measure how concentrated positive historical excess is in the best five months."""
    required = ["original_d11", "repaired_r0", "r1_nominal"]
    if not set(["month", *required]).issubset(monthly_three_way):
        raise DiagnosticError("Day-6 monthly three-way table is incomplete")
    rows = []
    for strategy in required:
        values = pd.to_numeric(monthly_three_way[strategy], errors="coerce").dropna()
        positive = values.clip(lower=0)
        top = values.nlargest(min(5, len(values)))
        best = values.idxmax()
        rows.append({
            "strategy": strategy, "months": int(len(values)),
            "best_month": str(monthly_three_way.loc[best, "month"]),
            "best_month_excess": float(values.loc[best]),
            "best_five_months_excess_sum": float(top.sum()),
            "best_five_share_of_positive_excess": (
                float(top.clip(lower=0).sum() / positive.sum()) if positive.sum() > 0 else None),
            "full_period_monthly_excess_sum": float(values.sum()),
        })
    return pd.DataFrame(rows)


def evidence_matrix(day5_summary):
    """Keep repair effects, strategy effects, stress paths and unrun R2 visibly separate."""
    performance = day5_summary["performance"]
    labels = [
        ("original_d11_reference", "historical reference; not strategy improvement"),
        ("repaired_r0_nominal", "price repair effect belongs to baseline repair"),
        ("r1_nominal", "R1 strategy comparison versus repaired R0"),
        ("repaired_r0_impact2x", "execution stress; actual cash/holdings feedback"),
        ("r1_impact2x", "execution stress; decision k remains 0.5"),
    ]
    rows = []
    for name, interpretation in labels:
        item = performance[name]
        row = {"scenario": name, "status": "completed"}
        row.update({key: value for key, value in item.items()
                    if not isinstance(value, (dict, list, tuple))})
        row.update({f"newey_west_t_lag_{lag}": value
                    for lag, value in item.get("newey_west_t_by_lag", {}).items()})
        row["interpretation"] = interpretation
        rows.append(row)
    rows.append({"scenario": "r2", "status": "not_run",
                 "interpretation": "frozen for next week; start conditions not met"})
    return pd.DataFrame(rows)


def r2_readiness(unresolved, *, day5_completed, p2_manual_review_completed=False,
                 resource_budget_verified=False):
    """Evaluate pre-registered start conditions without silently launching R2."""
    unresolved = list(unresolved)
    rows = [
        {"condition": "R0 credible; no unresolved class-A P0",
         "status": "blocked" if unresolved else "passed",
         "evidence": ", ".join(unresolved) if unresolved else "none"},
        {"condition": "R1 nominal and execution-impact stress completed",
         "status": "passed" if day5_completed else "blocked",
         "evidence": "day5 completed manifest" if day5_completed else "day5 incomplete"},
        {"condition": "P2 review found no uncorrected data/calibration error",
         "status": "passed" if p2_manual_review_completed else "pending_manual_review",
         "evidence": "automatic screens are descriptive and cannot close this condition"},
        {"condition": "resource budget leaves time for complete run and review",
         "status": "passed" if resource_budget_verified else "unverified",
         "evidence": "R2 training runtime has not been measured"},
    ]
    table = pd.DataFrame(rows)
    ready = bool(table.status.eq("passed").all())
    return table, {
        "ready_to_start": ready,
        "decision": "eligible_to_start" if ready else "not_started_blocked",
        "next_review_date": "2026-09-16",
        "r2_run_this_week": False,
    }


def forward_startup_checklist(experiments):
    """Render the frozen forward spec as an honest not-yet-deployed checklist."""
    validate_day6_freeze(experiments)
    forward = experiments["day6_freeze"]["forward_recording"]
    rows = [
        ("freeze minimum specification", "planned", forward["spec_freeze_target"]),
        ("new-date market/calendar refresh", "pending", "must extend beyond 2026-08-31"),
        ("point-in-time membership evidence", "blocked", "P0-12 remains unresolved"),
        ("feature warm-up and causal calibration", "pending", "verify before first signal"),
        ("forward-valid tax/cost configuration", "blocked", "add separate version before fills after 2026-08-31"),
        ("R0 score and simulated-order log schema", "specified", ", ".join(forward["minimum_fields"])),
        ("R1 independent shadow path", "optional", "never choose the better path day by day"),
        ("forward logging start", "planned", forward["planned_start"]),
        ("first operating/signal review", "planned", forward["first_review_target"]),
    ]
    return pd.DataFrame(rows, columns=["item", "status", "evidence_or_requirement"])

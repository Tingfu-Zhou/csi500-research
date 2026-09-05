from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.d7_core import (
    OptimizationLimits,
    OptimizationResult,
    PortfolioOptimizationError,
    audit_portfolio_constraints,
)


class OOSIntegrityError(RuntimeError):
    """Raised when the frozen one-shot OOS contract is violated."""


def validate_v2_source_failures(
    original_manifest: Mapping[str, object],
    recovery_manifest: Mapping[str, object],
    *,
    expected_original: Mapping[str, object],
    expected_recovery: Mapping[str, object],
) -> dict[str, object]:
    """Require byte-registered D9 failures before a post-OOS V2 may run."""

    mismatches: dict[str, dict[str, object]] = {}
    for prefix, manifest, expected in [
        ("original", original_manifest, expected_original),
        ("recovery", recovery_manifest, expected_recovery),
    ]:
        for key, value in expected.items():
            if manifest.get(key) != value:
                mismatches[f"{prefix}.{key}"] = {
                    "expected": value,
                    "actual": manifest.get(key),
                }
    if mismatches:
        raise OOSIntegrityError(
            "D9 V2 来源失败清单与登记不一致：" + str(mismatches)
        )
    recovery_traceback = str(recovery_manifest.get("traceback", ""))
    required_traceback_fragments = [
        "generate_final_targets",
        "optimize_long_only_active_portfolio",
        "PortfolioOptimizationError",
        "status=infeasible",
    ]
    missing = [
        value for value in required_traceback_fragments
        if value not in recovery_traceback
    ]
    if missing:
        raise OOSIntegrityError(
            "D9 V2 恢复失败 traceback 缺少登记片段：" + ", ".join(missing)
        )
    return {
        "validated": True,
        "original_status": str(original_manifest["status"]),
        "recovery_status": str(recovery_manifest["status"]),
        "rank_ic_observed_before_v2": True,
        "portfolio_performance_observed_before_v2": False,
        "classification": "post_oos_methodology_amendment",
    }


def _prepare_turnover_v2_inputs(
    expected_excess_return: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    limits: OptimizationLimits,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    limits.validate()

    def numeric(values: pd.Series, name: str) -> pd.Series:
        result = pd.to_numeric(values, errors="coerce").astype(float)
        if result.index.has_duplicates:
            raise ValueError(f"{name} 存在重复索引")
        if result.isna().any() or not np.isfinite(result.to_numpy()).all():
            raise ValueError(f"{name} 含缺失或非有限值")
        return result

    benchmark = numeric(benchmark_weights, "benchmark_weights")
    pretrade = numeric(pretrade_weights, "pretrade_weights")
    mu = numeric(expected_excess_return, "expected_excess_return")
    if len(benchmark) < 2:
        raise ValueError("D9 V2 至少需要两只候选股票")
    if (benchmark < 0).any() or (pretrade < 0).any():
        raise ValueError("基准与调仓前权重不能为负")
    if not np.isclose(benchmark.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("基准权重和必须为 1")
    if not np.isclose(pretrade.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("调仓前权重和必须为 1")
    mu = mu.reindex(benchmark.index)
    if mu.isna().any():
        missing = mu.index[mu.isna()].tolist()
        raise ValueError("预期收益缺少股票：" + ", ".join(map(str, missing[:20])))
    if exposures.index.has_duplicates or set(exposures.index) != set(benchmark.index):
        raise ValueError("因子暴露与基准股票集合不一致")
    frame = exposures.reindex(benchmark.index).copy()
    industries = [str(value) for value in industry_columns]
    styles = [str(value) for value in style_columns]
    columns = industries + styles
    if len(set(columns)) != len(columns):
        raise ValueError("行业与风格约束列存在重复")
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError("因子暴露缺少列：" + ", ".join(missing_columns))
    factors = frame[columns].apply(pd.to_numeric, errors="coerce")
    if factors.isna().any().any() or not np.isfinite(factors.to_numpy()).all():
        raise ValueError("因子暴露含缺失或非有限值")
    frame[columns] = factors
    return mu, benchmark, pretrade, frame, industries, styles


def _solve_turnover_v2_problem(
    expected_excess_return: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    *,
    cost_rate: float,
    limits: OptimizationLimits,
    solver: str,
    minimize_turnover: bool,
) -> tuple[OptimizationResult, dict[str, object]]:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - runtime packaging
        raise PortfolioOptimizationError(
            "D9 V2 需要 cvxpy 与 Clarabel"
        ) from exc
    if str(solver).upper() != "CLARABEL":
        raise ValueError("D9 V2 只允许使用 CLARABEL")
    if "CLARABEL" not in cp.installed_solvers():
        raise PortfolioOptimizationError("当前 cvxpy 环境没有可用的 CLARABEL")
    rate = float(cost_rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("目标函数成本系数必须是非负有限数")
    mu, benchmark, pretrade, frame, industries, styles = (
        _prepare_turnover_v2_inputs(
            expected_excess_return,
            benchmark_weights,
            pretrade_weights,
            exposures,
            industry_columns,
            style_columns,
            limits,
        )
    )
    index = benchmark.index
    wb = benchmark.to_numpy(float)
    mu_values = mu.to_numpy(float)
    w0_current = pretrade.reindex(index, fill_value=0.0).to_numpy(float)
    forced_exit = float(pretrade.loc[~pretrade.index.isin(index)].sum())
    variable = cp.Variable(len(index), name="d9_v2_portfolio_weight")
    active = variable - wb
    current_universe_l1 = cp.norm1(variable - w0_current)
    total_l1 = current_universe_l1 + forced_exit
    constraints = [
        cp.sum(variable) == 1.0,
        variable >= 0.0,
        active <= limits.stock_active,
        active >= -limits.stock_active,
    ]
    if industries:
        industry_active = frame[industries].to_numpy(float).T @ active
        constraints.extend(
            [
                industry_active <= limits.industry_active,
                industry_active >= -limits.industry_active,
            ]
        )
    if styles:
        style_active = frame[styles].to_numpy(float).T @ active
        constraints.extend(
            [
                style_active <= limits.style_active,
                style_active >= -limits.style_active,
            ]
        )
    if not minimize_turnover:
        # Fully invested portfolios satisfy current-universe L1 >= forced_exit.
        # Hence total single-sided turnover minus forced_exit is the optional
        # (alpha-driven) part of trading. Mandatory exits still enter total_l1
        # and therefore remain fully charged in the objective and backtest.
        constraints.append(
            0.5 * total_l1 <= forced_exit + limits.single_sided_turnover
        )
        objective = cp.Maximize(mu_values @ active - rate * total_l1)
    else:
        objective = cp.Minimize(total_l1)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except cp.error.SolverError as exc:
        raise PortfolioOptimizationError(f"D9 V2 CLARABEL 求解失败：{exc}") from exc
    status = str(problem.status)
    if status != str(cp.OPTIMAL) or variable.value is None:
        purpose = "最小换手可行性审计" if minimize_turnover else "组合优化"
        raise PortfolioOptimizationError(
            f"D9 V2 {purpose}未得到严格 optimal，status={status}"
        )
    raw = np.asarray(variable.value, dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise PortfolioOptimizationError("D9 V2 CLARABEL 返回非有限权重")
    cleaned = raw.copy()
    cleaned[np.abs(cleaned) <= limits.tolerance * 0.1] = 0.0
    if cleaned.min() < -limits.tolerance:
        raise PortfolioOptimizationError("D9 V2 CLARABEL 返回超过容差的负权重")
    cleaned = np.maximum(cleaned, 0.0)
    cleaned /= float(cleaned.sum())
    weights = pd.Series(cleaned, index=index, name="target_weight")
    adjusted_limits = OptimizationLimits(
        stock_active=limits.stock_active,
        industry_active=limits.industry_active,
        style_active=limits.style_active,
        single_sided_turnover=(
            1.0 if minimize_turnover
            else min(1.0, forced_exit + limits.single_sided_turnover)
        ),
        tolerance=limits.tolerance,
    )
    audit = audit_portfolio_constraints(
        weights,
        benchmark,
        pretrade,
        frame,
        industries,
        styles,
        adjusted_limits,
    )
    total_turnover = float(audit["single_sided_turnover"])
    discretionary = max(total_turnover - forced_exit, 0.0)
    audit["mandatory_membership_turnover"] = forced_exit
    audit["discretionary_single_sided_turnover"] = discretionary
    audit["frozen_discretionary_turnover_limit"] = float(
        limits.single_sided_turnover
    )
    audit["v2_total_turnover_budget"] = min(
        1.0, forced_exit + limits.single_sided_turnover
    )
    audit["checks"]["total_turnover_v2_budget"] = audit["checks"].pop(
        "single_sided_turnover"
    )
    audit["checks"]["discretionary_turnover"] = bool(
        discretionary <= limits.single_sided_turnover + limits.tolerance
    )
    audit["passed"] = bool(all(audit["checks"].values()))
    non_turnover_passed = bool(
        all(
            passed for name, passed in audit["checks"].items()
            if name not in {"total_turnover_v2_budget", "discretionary_turnover"}
        )
    )
    if (minimize_turnover and not non_turnover_passed) or (
        not minimize_turnover and not audit["passed"]
    ):
        failed = [
            name for name, passed in audit["checks"].items()
            if not passed
            and (not minimize_turnover or name not in {
                "total_turnover_v2_budget", "discretionary_turnover"
            })
        ]
        raise PortfolioOptimizationError(
            "D9 V2 求解结果未通过实际约束审计：" + ", ".join(failed)
        )
    active_values = weights.to_numpy(float) - wb
    expected_active = float(mu_values @ active_values)
    estimated_cost = rate * float(audit["total_l1_trade"])
    result = OptimizationResult(
        weights=weights,
        status=status,
        objective_value=(
            -float(audit["total_l1_trade"])
            if minimize_turnover
            else expected_active - estimated_cost
        ),
        expected_active_return=expected_active,
        estimated_linear_cost=estimated_cost,
        audit=audit,
    )
    feasibility = {
        "solver_status": status,
        "minimum_total_single_sided_turnover": total_turnover,
        "mandatory_membership_turnover": forced_exit,
        "minimum_discretionary_single_sided_turnover": discretionary,
        "frozen_total_turnover_limit": float(limits.single_sided_turnover),
        "v2_total_turnover_budget": min(
            1.0, forced_exit + limits.single_sided_turnover
        ),
        "frozen_rule_feasible": bool(
            total_turnover <= limits.single_sided_turnover + limits.tolerance
        ),
        "v2_rule_feasible": bool(
            discretionary <= limits.single_sided_turnover + limits.tolerance
        ),
        "frozen_turnover_shortfall": max(
            total_turnover - limits.single_sided_turnover, 0.0
        ),
        "non_turnover_constraints_passed": non_turnover_passed,
        "forced_exit_instrument_count": int(
            (~pretrade.index.isin(index)).sum()
        ),
        "forced_exit_instruments": {
            str(key): float(value)
            for key, value in pretrade.loc[~pretrade.index.isin(index)]
            .sort_values(ascending=False)
            .items()
        },
        "constraint_metrics": {
            "maximum_stock_active": float(audit["maximum_stock_active"]),
            "maximum_industry_active": float(audit["maximum_industry_active"]),
            "maximum_style_active": float(audit["maximum_style_active"]),
            "weight_sum_error": float(audit["weight_sum_error"]),
            "minimum_weight": float(audit["minimum_weight"]),
        },
    }
    return result, feasibility


def audit_minimum_turnover_feasibility(
    expected_excess_return: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    *,
    limits: OptimizationLimits,
    solver: str = "CLARABEL",
) -> dict[str, object]:
    """Minimize turnover under every frozen non-turnover constraint."""

    _, feasibility = _solve_turnover_v2_problem(
        expected_excess_return,
        benchmark_weights,
        pretrade_weights,
        exposures,
        industry_columns,
        style_columns,
        cost_rate=0.0,
        limits=limits,
        solver=solver,
        minimize_turnover=True,
    )
    return feasibility


def optimize_long_only_active_portfolio_v2(
    expected_excess_return: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    *,
    cost_rate: float,
    limits: OptimizationLimits,
    solver: str = "CLARABEL",
) -> OptimizationResult:
    """D9 V2: cap discretionary turnover; charge all mandatory trading."""

    result, _ = _solve_turnover_v2_problem(
        expected_excess_return,
        benchmark_weights,
        pretrade_weights,
        exposures,
        industry_columns,
        style_columns,
        cost_rate=cost_rate,
        limits=limits,
        solver=solver,
        minimize_turnover=False,
    )
    return result


def validate_membership_recovery_failure(
    manifest: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Authorize only the exact pre-return membership failure registered in config."""

    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise OOSIntegrityError(
            "D9 原始失败清单与受控恢复登记不一致：" + str(mismatches)
        )
    original_traceback = str(manifest.get("traceback", ""))
    if not original_traceback.strip():
        raise OOSIntegrityError("D9 原始失败清单缺少 traceback，不得受控恢复")
    if "build_full_benchmark_weights" not in original_traceback:
        raise OOSIntegrityError("D9 原始失败不在成分权重构建阶段，不得受控恢复")
    return {
        "validated": True,
        "status": manifest["status"],
        "oos_run_number": int(manifest["oos_run_number"]),
        "failed_d9_head_commit": str(manifest["d9_head_commit"]),
        "failed_config_sha256": str(manifest["config_sha256"]),
        "error_type": str(manifest["error_type"]),
        "error": str(manifest["error"]),
        "performance_observed_before_failure": False,
    }


def _interval_covers(
    intervals: pd.DataFrame,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    rows = intervals.loc[
        intervals["instrument"].astype(str).eq(instrument)
        & intervals["end_date"].ge(start)
        & intervals["start_date"].le(end),
        ["start_date", "end_date"],
    ].sort_values("start_date")
    cursor = start
    for row in rows.itertuples(index=False):
        left = max(pd.Timestamp(row.start_date).normalize(), start)
        right = min(pd.Timestamp(row.end_date).normalize(), end)
        if left > cursor:
            return False
        cursor = max(cursor, right + pd.Timedelta(days=1))
        if cursor > end:
            return True
    return cursor > end


def apply_identifier_alias_removals(
    intervals: pd.DataFrame,
    removals: Iterable[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove a duplicate historical identifier while retaining its canonical peer."""

    required = {"instrument", "start_date", "end_date"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"成分区间缺少字段：{sorted(missing)}")
    result = intervals.copy()
    result["start_date"] = pd.to_datetime(result["start_date"]).dt.normalize()
    result["end_date"] = pd.to_datetime(result["end_date"]).dt.normalize()
    audits: list[dict[str, object]] = []
    for number, removal in enumerate(removals, start=1):
        alias = str(removal["alias_instrument"])
        canonical = str(removal["canonical_instrument"])
        start = pd.Timestamp(removal["effective_start"]).normalize()
        end = pd.Timestamp(removal["effective_end"]).normalize()
        if alias == canonical or start > end:
            raise ValueError(f"第 {number} 条标识别名删除规则无效")
        if not _interval_covers(result, canonical, start, end):
            raise ValueError(
                f"规范标识 {canonical} 未完整覆盖 {start.date()} 至 {end.date()}"
            )
        mask = (
            result["instrument"].astype(str).eq(alias)
            & result["end_date"].ge(start)
            & result["start_date"].le(end)
        )
        affected = result.loc[mask].copy()
        if affected.empty:
            raise ValueError(f"标识别名 {alias} 在删除窗口内没有成分区间")
        kept = result.loc[~mask].copy()
        pieces: list[dict[str, object]] = []
        for row in affected.to_dict(orient="records"):
            if row["start_date"] < start:
                before = dict(row)
                before["end_date"] = start - pd.Timedelta(days=1)
                pieces.append(before)
            if row["end_date"] > end:
                after = dict(row)
                after["start_date"] = end + pd.Timedelta(days=1)
                pieces.append(after)
        if pieces:
            kept = pd.concat([kept, pd.DataFrame(pieces)], ignore_index=True)
        result = kept.sort_values(["instrument", "start_date", "end_date"]).reset_index(
            drop=True
        )
        audits.append(
            {
                "rule_number": number,
                "alias_instrument": alias,
                "canonical_instrument": canonical,
                "effective_start": str(start.date()),
                "effective_end": str(end.date()),
                "removed_interval_rows": int(len(affected)),
                "reason": str(removal.get("reason", "")),
                "evidence": str(removal.get("evidence", "")),
            }
        )
    return result, audits


def apply_snapshot_instrument_bridges(
    snapshots: pd.DataFrame,
    bridges: Iterable[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Bridge a later successor back to its predecessor in one synthetic snapshot."""

    result = snapshots.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    audits: list[dict[str, object]] = []
    for number, bridge in enumerate(bridges, start=1):
        date = pd.Timestamp(bridge["trade_date"]).normalize()
        source = str(bridge["source_instrument"])
        target = str(bridge["target_instrument"])
        on_date = result["trade_date"].eq(date)
        source_mask = on_date & result["instrument"].astype(str).eq(source)
        target_mask = on_date & result["instrument"].astype(str).eq(target)
        if int(source_mask.sum()) != 1 or target_mask.any():
            raise ValueError(
                f"第 {number} 条合成快照桥接要求来源恰好一条且目标不存在"
            )
        if "snapshot_origin" in result.columns:
            origin = str(result.loc[source_mask, "snapshot_origin"].iloc[0])
            if origin != "backcast_membership_correction":
                raise ValueError("合成快照桥接不得修改 Tushare 原始快照")
        result.loc[source_mask, "instrument"] = target
        audits.append(
            {
                "rule_number": number,
                "trade_date": str(date.date()),
                "source_instrument": source,
                "target_instrument": target,
                "weight_preserved": True,
                "reason": str(bridge.get("reason", "")),
                "evidence": str(bridge.get("evidence", "")),
            }
        )
    if result.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("合成快照桥接后出现重复股票")
    return result, audits


@dataclass(frozen=True)
class OOSThresholds:
    annualized_excess_return_min: float
    information_ratio_min: float
    newey_west_t_min: float
    maximum_relative_drawdown_max: float
    specific_contribution_share_min: float
    doubled_impact_annualized_excess_min: float

    @classmethod
    def from_config(cls, values: Mapping[str, object]) -> "OOSThresholds":
        result = cls(
            annualized_excess_return_min=float(
                values["annualized_excess_return_min"]
            ),
            information_ratio_min=float(values["information_ratio_min"]),
            newey_west_t_min=float(values["newey_west_t_min"]),
            maximum_relative_drawdown_max=float(
                values["maximum_relative_drawdown_max"]
            ),
            specific_contribution_share_min=float(
                values["specific_contribution_share_min"]
            ),
            doubled_impact_annualized_excess_min=float(
                values["doubled_impact_annualized_excess_min"]
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        values = np.asarray(
            [
                self.annualized_excess_return_min,
                self.information_ratio_min,
                self.newey_west_t_min,
                self.maximum_relative_drawdown_max,
                self.specific_contribution_share_min,
                self.doubled_impact_annualized_excess_min,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("D9 门槛必须全部为有限数")
        if self.maximum_relative_drawdown_max < 0:
            raise ValueError("最大相对回撤上限不能为负")
        if not 0 <= self.specific_contribution_share_min <= 1:
            raise ValueError("特异性贡献占比门槛必须位于 [0, 1]")


def validate_oos_fold_records(
    records: pd.DataFrame,
    *,
    oos_start,
    oos_end,
) -> dict[str, object]:
    """Require one contiguous calendar-year fold for every OOS year."""

    required = {
        "prediction_year",
        "train_start",
        "train_end",
        "prediction_start",
        "prediction_end",
    }
    missing = required - set(records.columns)
    if missing:
        raise OOSIntegrityError(f"D9 折记录缺少字段：{sorted(missing)}")
    frame = records.copy().sort_values("prediction_year").reset_index(drop=True)
    if frame.empty or frame["prediction_year"].duplicated().any():
        raise OOSIntegrityError("D9 折记录为空或预测年份重复")
    start = pd.Timestamp(oos_start).normalize()
    end = pd.Timestamp(oos_end).normalize()
    expected_years = list(range(start.year, end.year + 1))
    actual_years = frame["prediction_year"].astype(int).tolist()
    if actual_years != expected_years:
        raise OOSIntegrityError(
            f"D9 必须完整运行 {expected_years}，实际为 {actual_years}"
        )
    for column in ["train_start", "train_end", "prediction_start", "prediction_end"]:
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    for number, row in frame.iterrows():
        year = int(row["prediction_year"])
        expected_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        expected_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if row["prediction_start"] != expected_start or row["prediction_end"] != expected_end:
            raise OOSIntegrityError(f"{year} 折预测边界未严格取自 config.yaml")
        if row["train_end"] != expected_start - pd.Timedelta(days=1):
            raise OOSIntegrityError(f"{year} 折训练截止与预测起点不连续")
        if number and row["train_start"] != frame.loc[0, "train_start"]:
            raise OOSIntegrityError("D9 必须使用固定起点的扩展训练窗口")
    return {
        "folds": int(len(frame)),
        "years": actual_years,
        "start": str(start.date()),
        "end": str(end.date()),
        "passed": True,
    }


def newey_west_t_statistic(values: Iterable[float], *, lag: int) -> float:
    """Newey-West t statistic for a non-zero mean with Bartlett weights."""

    series = (
        pd.Series(list(values), dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    count = len(series)
    lag = int(lag)
    if lag < 0:
        raise ValueError("Newey-West lag 不能为负")
    if count <= lag + 1:
        raise ValueError(
            f"Newey-West 观察数 {count} 不足以支持 lag={lag}"
        )
    demeaned = series.to_numpy(float) - float(series.mean())
    gamma_zero = float(demeaned @ demeaned / count)
    long_run_variance = gamma_zero
    for offset in range(1, lag + 1):
        covariance = float(
            demeaned[offset:] @ demeaned[:-offset] / count
        )
        bartlett = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * bartlett * covariance
    variance_of_mean = long_run_variance / count
    if variance_of_mean < -1e-18:
        raise ValueError("Newey-West 长期方差为负")
    if variance_of_mean <= 1e-18:
        mean = float(series.mean())
        if abs(mean) <= 1e-15:
            return 0.0
        return float(np.copysign(np.inf, mean))
    return float(series.mean() / np.sqrt(variance_of_mean))


def aggregate_daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound portfolio and benchmark daily returns before differencing."""

    required = {"trade_date", "portfolio_return", "benchmark_return"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"D9 日收益缺少字段：{sorted(missing)}")
    frame = daily[list(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if frame["trade_date"].duplicated().any():
        raise ValueError("D9 日收益存在重复日期")
    for column in ["portfolio_return", "benchmark_return"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["portfolio_return", "benchmark_return"]].isna().any().any():
        raise ValueError("D9 日收益含缺失或非数值")
    if (frame[["portfolio_return", "benchmark_return"]] <= -1.0).any().any():
        raise ValueError("D9 日收益不能低于或等于 -100%")
    frame = frame.sort_values("trade_date")
    frame["month"] = frame["trade_date"].dt.to_period("M")
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        portfolio = float((1.0 + group["portfolio_return"]).prod() - 1.0)
        benchmark = float((1.0 + group["benchmark_return"]).prod() - 1.0)
        rows.append(
            {
                "month": str(month),
                "month_end": group["trade_date"].max(),
                "portfolio_return": portfolio,
                "benchmark_return": benchmark,
                "excess_return": portfolio - benchmark,
                "trade_days": int(len(group)),
            }
        )
    if not rows:
        raise ValueError("D9 没有可聚合的月度收益")
    return pd.DataFrame(rows)


def maximum_relative_drawdown(daily: pd.DataFrame) -> float:
    """Maximum drawdown of portfolio wealth relative to benchmark wealth."""

    required = {"portfolio_return", "benchmark_return"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"相对回撤输入缺少字段：{sorted(missing)}")
    frame = daily[list(required)].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or (frame <= -1.0).any().any():
        raise ValueError("相对回撤输入收益无效")
    portfolio_wealth = (1.0 + frame["portfolio_return"]).cumprod()
    benchmark_wealth = (1.0 + frame["benchmark_return"]).cumprod()
    relative = np.concatenate(
        ([1.0], (portfolio_wealth / benchmark_wealth).to_numpy(float))
    )
    running_peak = np.maximum.accumulate(relative)
    drawdown = relative / running_peak - 1.0
    return float(-drawdown.min())


def summarize_monthly_performance(
    daily: pd.DataFrame,
    *,
    newey_west_lags: Iterable[int],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Calculate the D9 return statistics from monthly excess returns."""

    monthly = aggregate_daily_to_monthly(daily)
    excess = monthly["excess_return"]
    if len(excess) < 2:
        raise ValueError("D9 月度收益观察不足")
    annualized_excess = float(excess.mean() * 12.0)
    annualized_te = float(excess.std(ddof=1) * np.sqrt(12.0))
    if annualized_te > 0:
        information_ratio = float(annualized_excess / annualized_te)
    elif abs(annualized_excess) <= 1e-15:
        information_ratio = 0.0
    else:
        information_ratio = float(np.copysign(np.inf, annualized_excess))
    lags = tuple(sorted({int(value) for value in newey_west_lags}))
    if not lags or lags[0] < 0:
        raise ValueError("必须配置至少一个非负 Newey-West lag")
    t_values = {
        str(lag): newey_west_t_statistic(excess, lag=lag) for lag in lags
    }
    portfolio_cumulative = float((1.0 + monthly["portfolio_return"]).prod() - 1.0)
    benchmark_cumulative = float((1.0 + monthly["benchmark_return"]).prod() - 1.0)
    return (
        {
            "months": int(len(monthly)),
            "annualized_excess_return": annualized_excess,
            "annualized_tracking_error": annualized_te,
            "information_ratio": information_ratio,
            "identity_ir_times_te": float(information_ratio * annualized_te),
            "newey_west_t_by_lag": t_values,
            "newey_west_t_minimum": float(min(t_values.values())),
            "newey_west_decision_rule": "minimum_across_configured_lags",
            "maximum_relative_drawdown": maximum_relative_drawdown(daily),
            "portfolio_cumulative_return": portfolio_cumulative,
            "benchmark_cumulative_return": benchmark_cumulative,
            "cumulative_excess_return_difference": (
                portfolio_cumulative - benchmark_cumulative
            ),
        },
        monthly,
    )


def evaluate_oos_thresholds(
    performance: Mapping[str, object],
    *,
    doubled_impact_annualized_excess: float,
    thresholds: OOSThresholds,
    specific_contribution_share: float | None = None,
) -> dict[str, object]:
    """Evaluate pre-registered thresholds and preserve the D10 dependency."""

    thresholds.validate()
    values = {
        "annualized_excess_return": float(performance["annualized_excess_return"]),
        "information_ratio": float(performance["information_ratio"]),
        "newey_west_t": float(performance["newey_west_t_minimum"]),
        "maximum_relative_drawdown": float(
            performance["maximum_relative_drawdown"]
        ),
        "specific_contribution_share": (
            None
            if specific_contribution_share is None
            else float(specific_contribution_share)
        ),
        "doubled_impact_annualized_excess": float(
            doubled_impact_annualized_excess
        ),
    }
    checks: dict[str, bool | None] = {
        "annualized_excess_return": bool(
            values["annualized_excess_return"]
            >= thresholds.annualized_excess_return_min
        ),
        "information_ratio": bool(
            values["information_ratio"] >= thresholds.information_ratio_min
        ),
        "newey_west_t": bool(
            values["newey_west_t"] >= thresholds.newey_west_t_min
        ),
        "maximum_relative_drawdown": bool(
            values["maximum_relative_drawdown"]
            <= thresholds.maximum_relative_drawdown_max
        ),
        "specific_contribution_share": (
            None
            if values["specific_contribution_share"] is None
            else bool(
                values["specific_contribution_share"]
                >= thresholds.specific_contribution_share_min
            )
        ),
        "doubled_impact_positive": bool(
            values["doubled_impact_annualized_excess"]
            > thresholds.doubled_impact_annualized_excess_min
        ),
    }
    if checks["information_ratio"] is False:
        decision = "No-Go"
        reason = "样本外 IR 未达到预注册门槛"
    elif checks["newey_west_t"] is False:
        decision = "证据不足"
        reason = "IR 达标但 Newey-West t 值不足 2"
    elif any(value is False for value in checks.values()):
        decision = "No-Go"
        reason = "至少一项其余预注册门槛未通过"
    elif checks["specific_contribution_share"] is None:
        decision = "待D10归因"
        reason = "收益与稳健性门槛已评估；特异性贡献占比必须由 D10 WLS 归因给出"
    else:
        decision = "Go"
        reason = "六项预注册门槛全部通过"
    return {
        "values": values,
        "checks": checks,
        "decision": decision,
        "reason": reason,
        "all_six_available": checks["specific_contribution_share"] is not None,
        "all_six_passed": bool(
            checks["specific_contribution_share"] is not None
            and all(value is True for value in checks.values())
        ),
    }


def assert_frozen_config_unchanged(
    current_config: Mapping[str, object],
    frozen_config: Mapping[str, object],
) -> None:
    """Allow adding only the D9 section; every frozen key must remain identical."""

    current = dict(current_config)
    current.pop("d9", None)
    if current != dict(frozen_config):
        raise OOSIntegrityError(
            "当前 config.yaml 除新增 d9 节外与 frozen-oos 不一致"
        )

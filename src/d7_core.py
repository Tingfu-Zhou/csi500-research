from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class PortfolioOptimizationError(RuntimeError):
    """Raised when the D7 optimization contract cannot be satisfied."""


@dataclass(frozen=True)
class OptimizationLimits:
    stock_active: float
    industry_active: float
    style_active: float
    single_sided_turnover: float
    tolerance: float = 1e-6

    def validate(self) -> None:
        values = {
            "stock_active": self.stock_active,
            "industry_active": self.industry_active,
            "style_active": self.style_active,
            "single_sided_turnover": self.single_sided_turnover,
            "tolerance": self.tolerance,
        }
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("D7 约束上限与容差必须是非负有限数")
        if self.stock_active > 1 or self.industry_active > 1:
            raise ValueError("个股与行业主动偏离上限不能超过 1")
        if self.single_sided_turnover > 1:
            raise ValueError("单边换手上限不能超过 1")


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    objective_value: float
    expected_active_return: float
    estimated_linear_cost: float
    audit: dict[str, object]


@dataclass(frozen=True)
class TrackingErrorEstimate:
    annualized_tracking_error: float
    active_l1_coverage: float
    included_instruments: int
    excluded_instruments: tuple[str, ...]


def symmetric_l1_cost_rate(
    commission_rate: float,
    transfer_fee_rate: float,
    stamp_duty_rate: float,
) -> float:
    """Map fully-invested buy/sell fees to one coefficient on total L1 trades.

    With no external cash flow, total buys equal total sells.  Therefore a
    sell-only stamp duty is exactly represented by half its rate on every unit
    of L1 trade, rather than by silently treating buys as stampable.
    """

    rates = np.asarray(
        [commission_rate, transfer_fee_rate, stamp_duty_rate], dtype=float
    )
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("交易费率必须是非负有限数")
    return float(rates[0] + rates[1] + 0.5 * rates[2])


def drift_pretrade_weights(
    target_weights: pd.Series,
    holding_period_returns: pd.Series,
    *,
    weight_tolerance: float = 1e-12,
) -> pd.Series:
    """Drift the prior target to the next execution instant without look-ahead."""

    target = _numeric_series(target_weights, "target_weights")
    if target.index.has_duplicates:
        raise ValueError("target_weights 存在重复股票")
    if (target < -weight_tolerance).any() or not np.isclose(
        target.sum(), 1.0, atol=weight_tolerance, rtol=0.0
    ):
        raise ValueError("上一期目标权重必须非负且权重和为 1")
    returns = _numeric_series(holding_period_returns, "holding_period_returns")
    relevant = target[target > weight_tolerance]
    aligned = returns.reindex(relevant.index)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise PortfolioOptimizationError(
            "持仓漂移缺少收益：" + ", ".join(map(str, missing[:20]))
        )
    if (aligned < -1.0).any():
        raise ValueError("持仓期收益不能低于 -100%")
    gross = relevant * (1.0 + aligned)
    total = float(gross.sum())
    if not np.isfinite(total) or total <= 0:
        raise PortfolioOptimizationError("持仓漂移后的组合价值无效")
    result = (gross / total).sort_index()
    result.name = "pretrade_weight"
    return result


def audit_portfolio_constraints(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    limits: OptimizationLimits,
) -> dict[str, object]:
    """Measure every D7 hard constraint on the returned weights."""

    limits.validate()
    w, benchmark, pretrade, frame, industries, styles = _validate_inputs(
        weights,
        benchmark_weights,
        pretrade_weights,
        exposures,
        industry_columns,
        style_columns,
        require_weight_sum=True,
    )
    active = w - benchmark
    forced_exit = float(pretrade.loc[~pretrade.index.isin(w.index)].sum())
    current_pretrade = pretrade.reindex(w.index, fill_value=0.0)
    total_l1 = float((w - current_pretrade).abs().sum() + forced_exit)
    industry_exposure = frame[industries].T.dot(active)
    style_exposure = frame[styles].T.dot(active)
    tolerance = float(limits.tolerance)
    checks = {
        "fully_invested": bool(abs(float(w.sum()) - 1.0) <= tolerance),
        "long_only": bool(float(w.min()) >= -tolerance),
        "stock_active": bool(
            float(active.abs().max()) <= limits.stock_active + tolerance
        ),
        "industry_active": bool(
            _max_abs(industry_exposure) <= limits.industry_active + tolerance
        ),
        "style_active": bool(
            _max_abs(style_exposure) <= limits.style_active + tolerance
        ),
        "single_sided_turnover": bool(
            0.5 * total_l1 <= limits.single_sided_turnover + tolerance
        ),
    }
    return {
        "weight_sum": float(w.sum()),
        "weight_sum_error": abs(float(w.sum()) - 1.0),
        "minimum_weight": float(w.min()),
        "maximum_stock_active": float(active.abs().max()),
        "maximum_industry_active": _max_abs(industry_exposure),
        "maximum_style_active": _max_abs(style_exposure),
        "total_l1_trade": total_l1,
        "single_sided_turnover": 0.5 * total_l1,
        "forced_exit_weight": forced_exit,
        "industry_active_exposures": {
            str(key): float(value) for key, value in industry_exposure.items()
        },
        "style_active_exposures": {
            str(key): float(value) for key, value in style_exposure.items()
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def optimize_long_only_active_portfolio(
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
    """Solve the no-fallback D7 LP with costs in the objective."""

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - depends on runtime packaging
        raise PortfolioOptimizationError(
            "D7 需要 cvxpy 与 Clarabel；请先安装运行说明中的依赖"
        ) from exc

    limits.validate()
    mu = _numeric_series(expected_excess_return, "expected_excess_return")
    benchmark = _numeric_series(benchmark_weights, "benchmark_weights")
    pretrade = _numeric_series(pretrade_weights, "pretrade_weights")
    placeholder = benchmark.copy()
    placeholder.name = "candidate_weight"
    _, benchmark, pretrade, frame, industries, styles = _validate_inputs(
        placeholder,
        benchmark,
        pretrade,
        exposures,
        industry_columns,
        style_columns,
        require_weight_sum=True,
    )
    mu = mu.reindex(benchmark.index)
    if mu.isna().any():
        missing = mu.index[mu.isna()].tolist()
        raise ValueError("预期收益缺少股票：" + ", ".join(map(str, missing[:20])))
    rate = float(cost_rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("目标函数成本系数必须是非负有限数")
    if str(solver).upper() != "CLARABEL":
        raise ValueError("D7 按整体计划只允许使用 CLARABEL")
    if "CLARABEL" not in cp.installed_solvers():
        raise PortfolioOptimizationError("当前 cvxpy 环境没有可用的 CLARABEL")

    index = benchmark.index
    n_assets = len(index)
    w0_current = pretrade.reindex(index, fill_value=0.0).to_numpy(float)
    forced_exit = float(pretrade.loc[~pretrade.index.isin(index)].sum())
    wb = benchmark.to_numpy(float)
    mu_values = mu.to_numpy(float)
    industry_matrix = frame[industries].to_numpy(float)
    style_matrix = frame[styles].to_numpy(float)

    variable = cp.Variable(n_assets, name="portfolio_weight")
    active = variable - wb
    total_l1 = cp.norm1(variable - w0_current) + forced_exit
    constraints = [
        cp.sum(variable) == 1.0,
        variable >= 0.0,
        active <= limits.stock_active,
        active >= -limits.stock_active,
        0.5 * total_l1 <= limits.single_sided_turnover,
    ]
    if industries:
        industry_active = industry_matrix.T @ active
        constraints.extend(
            [
                industry_active <= limits.industry_active,
                industry_active >= -limits.industry_active,
            ]
        )
    if styles:
        style_active = style_matrix.T @ active
        constraints.extend(
            [
                style_active <= limits.style_active,
                style_active >= -limits.style_active,
            ]
        )
    objective = cp.Maximize(mu_values @ active - rate * total_l1)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except cp.error.SolverError as exc:
        raise PortfolioOptimizationError(f"CLARABEL 求解失败：{exc}") from exc
    status = str(problem.status)
    if status != str(cp.OPTIMAL):
        raise PortfolioOptimizationError(
            f"CLARABEL 未得到严格 optimal，status={status}；D7 禁止 fallback"
        )
    if variable.value is None:
        raise PortfolioOptimizationError("CLARABEL 返回 optimal 但没有组合权重")

    raw = np.asarray(variable.value, dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise PortfolioOptimizationError("CLARABEL 返回非有限权重")
    cleaned = raw.copy()
    cleaned[np.abs(cleaned) <= limits.tolerance * 0.1] = 0.0
    if cleaned.min() < -limits.tolerance:
        raise PortfolioOptimizationError("CLARABEL 返回超过容差的负权重")
    cleaned = np.maximum(cleaned, 0.0)
    weight_sum = float(cleaned.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise PortfolioOptimizationError("CLARABEL 返回的权重和无效")
    cleaned /= weight_sum
    weights = pd.Series(cleaned, index=index, name="target_weight")
    audit = audit_portfolio_constraints(
        weights,
        benchmark,
        pretrade,
        frame,
        industries,
        styles,
        limits,
    )
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise PortfolioOptimizationError(
            "求解结果未通过实际约束审计：" + ", ".join(failed)
        )
    active_values = weights.to_numpy(float) - wb
    expected_active = float(mu_values @ active_values)
    estimated_cost = rate * float(audit["total_l1_trade"])
    return OptimizationResult(
        weights=weights,
        status=status,
        objective_value=expected_active - estimated_cost,
        expected_active_return=expected_active,
        estimated_linear_cost=estimated_cost,
        audit=audit,
    )


def annualized_tracking_error(
    active_weights: pd.Series,
    covariance: np.ndarray,
    covariance_instruments: Iterable[str],
    *,
    annualization_factor: float = 252.0,
) -> TrackingErrorEstimate:
    """Calculate report-only ex-ante TE and disclose covariance coverage."""

    active = _numeric_series(active_weights, "active_weights")
    instruments = pd.Index([str(item) for item in covariance_instruments])
    if instruments.has_duplicates:
        raise ValueError("协方差股票列表存在重复项")
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (len(instruments), len(instruments)):
        raise ValueError("协方差矩阵维度与股票列表不一致")
    if not np.isfinite(matrix).all():
        raise ValueError("协方差矩阵含非有限值")
    factor = float(annualization_factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("TE 年化因子必须为正的有限数")
    vector = active.reindex(instruments, fill_value=0.0).to_numpy(float)
    variance = float(vector @ matrix @ vector)
    if variance < -1e-12:
        raise ValueError("主动组合方差为负，协方差输入可能无效")
    variance = max(variance, 0.0)
    total_l1 = float(active.abs().sum())
    included_l1 = float(active.reindex(instruments, fill_value=0.0).abs().sum())
    coverage = 1.0 if total_l1 <= 1e-15 else included_l1 / total_l1
    excluded = tuple(sorted(set(active.index.astype(str)) - set(instruments)))
    return TrackingErrorEstimate(
        annualized_tracking_error=float(np.sqrt(variance * factor)),
        active_l1_coverage=float(coverage),
        included_instruments=len(instruments),
        excluded_instruments=excluded,
    )


def _numeric_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if result.index.has_duplicates:
        raise ValueError(f"{name} 存在重复索引")
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} 含缺失或非有限值")
    return result


def _validate_inputs(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    pretrade_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
    *,
    require_weight_sum: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    w = _numeric_series(weights, "weights")
    benchmark = _numeric_series(benchmark_weights, "benchmark_weights")
    pretrade = _numeric_series(pretrade_weights, "pretrade_weights")
    if benchmark.index.has_duplicates or pretrade.index.has_duplicates:
        raise ValueError("基准或调仓前权重存在重复股票")
    if len(benchmark) < 2:
        raise ValueError("D7 至少需要两只候选股票")
    if set(w.index) != set(benchmark.index):
        raise ValueError("组合与基准股票集合不一致")
    w = w.reindex(benchmark.index)
    if (benchmark < 0).any() or (pretrade < 0).any():
        raise ValueError("基准与调仓前权重不能为负")
    if not np.isclose(benchmark.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("基准权重和必须为 1")
    if not np.isclose(pretrade.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("调仓前权重和必须为 1")
    if require_weight_sum and not np.isclose(w.sum(), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("候选组合权重和必须为 1")
    if exposures.index.has_duplicates:
        raise ValueError("因子暴露存在重复股票")
    if set(exposures.index) != set(benchmark.index):
        raise ValueError("因子暴露与基准股票集合不一致")
    frame = exposures.reindex(benchmark.index).copy()
    industries = [str(item) for item in industry_columns]
    styles = [str(item) for item in style_columns]
    if len(set(industries + styles)) != len(industries) + len(styles):
        raise ValueError("行业与风格约束列存在重复")
    missing = set(industries + styles) - set(frame.columns)
    if missing:
        raise ValueError("因子暴露缺少列：" + ", ".join(sorted(missing)))
    factor_values = frame[industries + styles].apply(pd.to_numeric, errors="coerce")
    if factor_values.isna().any().any() or not np.isfinite(
        factor_values.to_numpy(float)
    ).all():
        raise ValueError("因子暴露含缺失或非有限值")
    frame[industries + styles] = factor_values
    return w, benchmark, pretrade, frame, industries, styles


def _max_abs(values: pd.Series) -> float:
    return float(values.abs().max()) if len(values) else 0.0

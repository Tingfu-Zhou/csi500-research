from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class LossLadderError(RuntimeError):
    """Raised when the D8 loss-ladder contract cannot be satisfied."""


LAYER_NAMES = {
    0: "Top100等权_无成本无约束",
    1: "加真实成本",
    2: "加可交易性过滤",
    3: "加行业中性",
    4: "加SizeBeta中性",
    5: "加换手上限",
    6: "加个股偏离上限",
}


LAYER_EXPLANATIONS = {
    0: "原始 Top100 等权信号基线",
    1: "在相同理想成交路径上加入动态冲击、佣金、过户费和卖出印花税",
    2: "启用停牌、涨跌停、ST、上市时长、流动性过滤及5%成交额上限",
    3: "组合构造切换为带成本目标的优化器，并加入行业主动偏离约束",
    4: "在行业约束上增加 Size 与 Beta 主动暴露约束",
    5: "相对同一冻结D7事前持仓，在上一层增加15%单边目标换手上限",
    6: "在上一层增加1.5%个股主动权重偏离上限；直接复用冻结的D7目标",
}


@dataclass(frozen=True)
class TargetExposureAudit:
    maximum_stock_active: float
    maximum_industry_active: float
    maximum_style_active: float
    industry_active_exposures: dict[str, float]
    style_active_exposures: dict[str, float]


def validate_layer_ids(layer_ids: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(item) for item in layer_ids)
    if result != tuple(range(7)):
        raise ValueError("D8 层级必须严格为 0–6，且顺序不能改变")
    return result


def make_top_n_equal_targets(
    predictions: pd.DataFrame,
    *,
    top_n: int,
    score_col: str,
    date_col: str = "sample_date",
) -> pd.DataFrame:
    """Select Top-N using deterministic score/instrument ordering per date."""

    required = {date_col, "instrument", score_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Top-N 输入缺少字段：{sorted(missing)}")
    count = int(top_n)
    if count < 1:
        raise ValueError("top_n 必须为正整数")
    frame = predictions[list(required)].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    if frame.duplicated([date_col, "instrument"]).any():
        raise ValueError("Top-N 输入存在重复日期-股票")
    if frame[score_col].isna().any() or not np.isfinite(
        frame[score_col].to_numpy(float)
    ).all():
        raise ValueError("Top-N 分数含缺失或非有限值")
    sizes = frame.groupby(date_col)["instrument"].size()
    if (sizes < count).any():
        bad_date = pd.Timestamp(sizes.index[sizes.argmin()]).date()
        raise ValueError(f"{bad_date} 的有效预测不足 Top {count}")
    selected = (
        frame.sort_values(
            [date_col, score_col, "instrument"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby(date_col, sort=True, group_keys=False)
        .head(count)
        .rename(columns={date_col: "signal_date"})
    )
    selected["target_weight"] = 1.0 / count
    return selected[
        ["signal_date", "instrument", "target_weight", score_col]
    ].reset_index(drop=True)


def audit_target_exposures(
    target_weights: pd.Series,
    benchmark_weights: pd.Series,
    exposures: pd.DataFrame,
    industry_columns: Iterable[str],
    style_columns: Iterable[str],
) -> TargetExposureAudit:
    target = _numeric_series(target_weights, "target_weights")
    benchmark = _numeric_series(benchmark_weights, "benchmark_weights")
    if not np.isclose(target.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("目标权重和必须为 1")
    if not np.isclose(benchmark.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("基准权重和必须为 1")
    if (target < -1e-12).any() or (benchmark < 0).any():
        raise ValueError("目标与基准权重不能为负")
    if set(target.index) - set(benchmark.index):
        raise ValueError("目标中含非基准成分股")
    target = target.reindex(benchmark.index, fill_value=0.0)
    if exposures.index.has_duplicates or set(exposures.index) != set(benchmark.index):
        raise ValueError("暴露矩阵股票集合与基准不一致")
    frame = exposures.reindex(benchmark.index)
    industries = [str(item) for item in industry_columns]
    styles = [str(item) for item in style_columns]
    missing = set(industries + styles) - set(frame.columns)
    if missing:
        raise ValueError("暴露矩阵缺少列：" + ", ".join(sorted(missing)))
    factor_values = frame[industries + styles].apply(pd.to_numeric, errors="coerce")
    if factor_values.isna().any().any():
        raise ValueError("暴露矩阵含缺失值")
    active = target - benchmark
    industry_active = factor_values[industries].T.dot(active)
    style_active = factor_values[styles].T.dot(active)
    return TargetExposureAudit(
        maximum_stock_active=float(active.abs().max()),
        maximum_industry_active=_max_abs(industry_active),
        maximum_style_active=_max_abs(style_active),
        industry_active_exposures={
            str(key): float(value) for key, value in industry_active.items()
        },
        style_active_exposures={
            str(key): float(value) for key, value in style_active.items()
        },
    )


def summarize_backtest_performance(
    nav: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_capital: float,
    annualization_factor: float = 252.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize from the first invested day and retain initial trading cost."""

    required_nav = {
        "trade_date",
        "nav",
        "gross_exposure",
        "single_sided_turnover",
        "transaction_cost",
        "signal_date",
    }
    required_benchmark = {"trade_date", "close"}
    if required_nav - set(nav.columns):
        raise ValueError(f"回测净值缺少字段：{sorted(required_nav - set(nav.columns))}")
    if required_benchmark - set(benchmark.columns):
        raise ValueError(
            f"基准输入缺少字段：{sorted(required_benchmark - set(benchmark.columns))}"
        )
    capital = float(initial_capital)
    annualization = float(annualization_factor)
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("初始资金必须为正")
    if not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("年化因子必须为正")
    portfolio = nav.copy()
    index = benchmark.copy()
    portfolio["trade_date"] = pd.to_datetime(portfolio["trade_date"]).dt.normalize()
    index["trade_date"] = pd.to_datetime(index["trade_date"]).dt.normalize()
    invested = np.flatnonzero(
        pd.to_numeric(portfolio["gross_exposure"], errors="coerce")
        .gt(0)
        .to_numpy()
    )
    if len(invested) == 0:
        raise LossLadderError("组合从未建仓")
    portfolio = portfolio.iloc[int(invested[0]) :].copy()
    merged = portfolio.merge(
        index[["trade_date", "close"]],
        on="trade_date",
        how="inner",
        validate="one_to_one",
    ).sort_values("trade_date")
    if len(merged) < 2:
        raise LossLadderError("组合与基准重叠交易日不足")
    merged["portfolio_return"] = merged["nav"].pct_change(fill_method=None)
    merged.iloc[0, merged.columns.get_loc("portfolio_return")] = (
        float(merged["nav"].iloc[0]) / capital - 1.0
    )
    merged["benchmark_return"] = merged["close"].pct_change(fill_method=None)
    merged.iloc[0, merged.columns.get_loc("benchmark_return")] = 0.0
    merged["active_return"] = (
        merged["portfolio_return"] - merged["benchmark_return"]
    )
    periods = len(merged) - 1
    portfolio_annualized = float(
        (float(merged["nav"].iloc[-1]) / capital) ** (annualization / periods) - 1.0
    )
    benchmark_annualized = float(
        (float(merged["close"].iloc[-1]) / float(merged["close"].iloc[0]))
        ** (annualization / periods)
        - 1.0
    )
    rebalance = merged.loc[merged["signal_date"].notna()]
    active_std = float(merged["active_return"].std(ddof=1))
    metrics = {
        "start_date": str(merged["trade_date"].min().date()),
        "end_date": str(merged["trade_date"].max().date()),
        "observations": int(len(merged)),
        "ending_nav": float(merged["nav"].iloc[-1]),
        "portfolio_annualized_return": portfolio_annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "annualized_excess_return": portfolio_annualized - benchmark_annualized,
        "realized_tracking_error": active_std * np.sqrt(annualization),
        "mean_rebalance_single_sided_turnover": float(
            rebalance["single_sided_turnover"].mean()
        ),
        "total_transaction_cost": float(merged["transaction_cost"].sum()),
        "transaction_cost_over_initial_capital": float(
            merged["transaction_cost"].sum() / capital
        ),
    }
    return metrics, merged.reset_index(drop=True)


def assert_layer6_matches_d7(
    ladder_targets: pd.DataFrame,
    d7_targets: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[str, float | int | bool]:
    required = {"signal_date", "instrument", "target_weight"}
    for name, frame in [("D8第6层", ladder_targets), ("D7", d7_targets)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}目标缺少字段：{sorted(missing)}")
        if frame.duplicated(["signal_date", "instrument"]).any():
            raise ValueError(f"{name}目标存在重复日期-股票")
    left = ladder_targets[list(required)].copy()
    right = d7_targets[list(required)].copy()
    for frame in [left, right]:
        frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    merged = left.merge(
        right,
        on=["signal_date", "instrument"],
        how="outer",
        suffixes=("_d8", "_d7"),
        indicator=True,
    )
    same_keys = bool(merged["_merge"].eq("both").all())
    difference = (
        merged["target_weight_d8"] - merged["target_weight_d7"]
        if same_keys
        else pd.Series([np.inf])
    )
    maximum = float(difference.abs().max())
    passed = bool(same_keys and maximum <= float(tolerance))
    if not passed:
        raise LossLadderError(
            f"D8 第6层未精确复用冻结 D7 目标；最大权重差={maximum:.3e}"
        )
    return {
        "rows": int(len(merged)),
        "same_keys": same_keys,
        "maximum_absolute_weight_difference": maximum,
        "passed": passed,
    }


def require_d7_structural_completion(summary: dict) -> dict[str, object]:
    """Allow the known TE miss, but reject any D7 engineering failure."""

    if summary.get("mode") != "design" or summary.get("oos_data_accessed") is not False:
        raise LossLadderError("D7 不是封存设计窗口产物")
    acceptance = summary.get("acceptance", {})
    structural = [
        "upstream_d6",
        "all_available_design_signal_dates",
        "solver_success_100pct",
        "fallback_zero",
        "fully_invested",
        "long_only",
        "stock_active",
        "industry_active",
        "style_active",
        "turnover",
        "covariance_coverage",
    ]
    failed = [key for key in structural if acceptance.get(key) is not True]
    failed.extend(
        key
        for key, value in acceptance.items()
        if value is False and key not in {"tracking_error", "passed"}
    )
    failed = sorted(set(failed))
    if failed:
        raise LossLadderError("D7 工程验收未完成：" + ", ".join(failed))
    return {
        "structural_checks_passed": True,
        "tracking_error_passed": bool(acceptance.get("tracking_error", False)),
        "overall_d7_passed": bool(acceptance.get("passed", False)),
    }


def build_loss_deltas(layer_summary: pd.DataFrame, *, capital: float) -> pd.DataFrame:
    required = {"capital", "layer", "annualized_excess_return"}
    missing = required - set(layer_summary.columns)
    if missing:
        raise ValueError(f"损耗汇总缺少字段：{sorted(missing)}")
    selected = layer_summary.loc[
        np.isclose(pd.to_numeric(layer_summary["capital"]), float(capital))
    ].sort_values("layer")
    if selected["layer"].tolist() != list(range(7)):
        raise ValueError("主资金规模必须完整包含第0–6层")
    result = selected[["layer", "annualized_excess_return"]].copy()
    result["previous_layer"] = result["layer"] - 1
    result["change_vs_previous"] = result["annualized_excess_return"].diff()
    result["explanation"] = result["layer"].map(LAYER_EXPLANATIONS)
    return result.reset_index(drop=True)


def make_ideal_execution_market(
    market: pd.DataFrame, *, fallback_daily_amount: float
) -> pd.DataFrame:
    """Remove tradability flags while causally filling amount for cost isolation."""

    required = {
        "trade_date",
        "instrument",
        "close",
        "vwap",
        "daily_amount_cny",
        "volatility",
    }
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"理想执行行情缺少字段：{sorted(missing)}")
    result = market[list(required)].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(["instrument", "trade_date"])
    amount = pd.to_numeric(result["daily_amount_cny"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    amount = amount.groupby(result["instrument"], sort=False).ffill()
    date_median = amount.groupby(result["trade_date"]).transform("median")
    fallback = float(fallback_daily_amount)
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("成交额回退值必须为正")
    result["daily_amount_cny"] = amount.fillna(date_median).fillna(fallback)
    result["can_buy"] = True
    result["can_sell"] = True
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def _numeric_series(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if result.index.has_duplicates:
        raise ValueError(f"{name} 存在重复索引")
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} 含缺失或非有限值")
    return result


def _max_abs(values: pd.Series) -> float:
    return float(values.abs().max()) if len(values) else 0.0

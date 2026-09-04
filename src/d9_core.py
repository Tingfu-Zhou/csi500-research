from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class OOSIntegrityError(RuntimeError):
    """Raised when the frozen one-shot OOS contract is violated."""


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

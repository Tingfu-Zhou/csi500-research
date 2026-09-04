from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.d3_core import make_label_windows


def select_weekly_sample_dates(benchmark_weights: pd.DataFrame) -> pd.DatetimeIndex:
    """Return each W-FRI week's last trading day represented in benchmark weights."""
    if "trade_date" not in benchmark_weights:
        raise ValueError("基准权重缺少 trade_date")
    dates = pd.Series(pd.to_datetime(benchmark_weights["trade_date"])).dt.normalize()
    unique = pd.Series(pd.DatetimeIndex(dates.unique()).sort_values())
    if unique.empty:
        raise ValueError("基准权重没有交易日")
    week = unique.dt.to_period("W-FRI")
    selected = unique.groupby(week).max()
    return pd.DatetimeIndex(selected.to_numpy()).sort_values()


def validate_alpha158_spec(
    fields: Sequence[str], names: Sequence[str], expected_count: int = 158
) -> None:
    """Validate the Qlib feature contract and reject explicit future Ref offsets."""
    if len(fields) != expected_count or len(names) != expected_count:
        raise ValueError(
            f"Alpha158 字段数异常：fields={len(fields)}, names={len(names)}, "
            f"期望={expected_count}"
        )
    if len(set(names)) != len(names):
        raise ValueError("Alpha158 特征名存在重复")
    future_ref = re.compile(r"Ref\s*\([^)]*?,\s*-\d+", flags=re.IGNORECASE)
    offenders = [field for field in fields if future_ref.search(str(field))]
    if offenders:
        raise ValueError(f"Alpha158 配置包含未来 Ref：{offenders[:3]}")


def _validate_label_inputs(
    vwap_prices: pd.DataFrame, benchmark_weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_required = {"trade_date", "instrument", "vwap"}
    weight_required = {"trade_date", "instrument", "weight"}
    if price_required - set(vwap_prices.columns):
        raise ValueError(f"VWAP 行情缺少字段：{sorted(price_required - set(vwap_prices.columns))}")
    if weight_required - set(benchmark_weights.columns):
        raise ValueError(
            f"基准权重缺少字段：{sorted(weight_required - set(benchmark_weights.columns))}"
        )

    prices = vwap_prices[list(price_required)].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.normalize()
    prices["vwap"] = pd.to_numeric(prices["vwap"], errors="coerce")
    if prices.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("VWAP 行情存在重复的 日期×股票")

    weights = benchmark_weights[list(weight_required)].copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"]).dt.normalize()
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    if weights.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("基准权重存在重复的 日期×股票")
    if weights["weight"].isna().any() or weights["weight"].lt(0).any():
        raise ValueError("基准权重包含缺失、非数值或负数")
    sums = weights.groupby("trade_date")["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-8):
        first = sums.loc[~np.isclose(sums, 1.0, rtol=0.0, atol=1e-8)].index[0]
        raise ValueError(f"{pd.Timestamp(first).date()} 基准权重和不等于 1")
    return prices, weights


def build_forward_excess_labels(
    vwap_prices: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    sample_dates: Iterable[pd.Timestamp],
    calendar: Iterable[pd.Timestamp],
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build T+1 VWAP to T+(h+1) VWAP constituent excess-return labels.

    The benchmark return is the signal-date w_b weighted return of constituents
    with usable endpoints, renormalized over those constituents. A stock without
    either endpoint has no label; its absence is audited but never used to drop
    an otherwise valid weekly cross-section.
    """
    requested_horizons = sorted({int(item) for item in horizons})
    if not requested_horizons or requested_horizons[0] <= 0:
        raise ValueError("标签周期必须是正整数")

    prices, weights = _validate_label_inputs(vwap_prices, benchmark_weights)
    requested_dates = pd.DatetimeIndex(pd.to_datetime(list(sample_dates))).normalize()
    requested_dates = requested_dates.sort_values().unique()
    base = weights.loc[weights["trade_date"].isin(requested_dates)].rename(
        columns={"trade_date": "sample_date", "weight": "benchmark_weight"}
    )
    if base.empty:
        raise ValueError("样本日与基准权重没有交集")

    label_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for horizon in requested_horizons:
        windows = make_label_windows(requested_dates, calendar, horizon)
        if windows.empty:
            raise ValueError(f"{horizon} 日标签没有完整时间窗")
        working = base.merge(windows, on="sample_date", how="inner", validate="many_to_one")
        entry = prices.rename(
            columns={"trade_date": "label_start_timestamp", "vwap": "entry_vwap"}
        )
        exit_ = prices.rename(
            columns={"trade_date": "label_end_timestamp", "vwap": "exit_vwap"}
        )
        working = working.merge(
            entry[["label_start_timestamp", "instrument", "entry_vwap"]],
            on=["label_start_timestamp", "instrument"],
            how="left",
            validate="many_to_one",
        ).merge(
            exit_[["label_end_timestamp", "instrument", "exit_vwap"]],
            on=["label_end_timestamp", "instrument"],
            how="left",
            validate="many_to_one",
        )
        valid_endpoint = working["entry_vwap"].gt(0) & working["exit_vwap"].gt(0)
        working["stock_return"] = np.where(
            valid_endpoint, working["exit_vwap"] / working["entry_vwap"] - 1.0, np.nan
        )
        working["available_weight"] = working["benchmark_weight"].where(
            working["stock_return"].notna(), 0.0
        )
        working["weighted_return"] = (
            working["available_weight"] * working["stock_return"].fillna(0.0)
        )
        grouped = working.groupby("sample_date", sort=False)
        coverage = grouped["available_weight"].transform("sum")
        benchmark_numerator = grouped["weighted_return"].transform("sum")
        working["weight_coverage"] = coverage
        working["benchmark_return"] = benchmark_numerator / coverage.where(coverage.gt(0))
        working["excess_return"] = working["stock_return"] - working["benchmark_return"]
        working["horizon"] = horizon

        audit = (
            working.groupby("sample_date", as_index=False)
            .agg(
                label_start_timestamp=("label_start_timestamp", "first"),
                label_end_timestamp=("label_end_timestamp", "first"),
                weight_coverage=("weight_coverage", "first"),
                constituent_count=("instrument", "size"),
                valid_label_count=("stock_return", "count"),
            )
            .assign(horizon=horizon)
        )
        coverage_frames.append(audit)
        usable = working["stock_return"].notna() & working["weight_coverage"].gt(0)
        label_frames.append(
            working.loc[
                usable,
                [
                    "sample_date",
                    "instrument",
                    "horizon",
                    "label_start_timestamp",
                    "label_end_timestamp",
                    "stock_return",
                    "benchmark_return",
                    "excess_return",
                    "benchmark_weight",
                    "weight_coverage",
                ],
            ]
        )

    labels = pd.concat(label_frames, ignore_index=True)
    coverage_audit = pd.concat(coverage_frames, ignore_index=True)
    if labels.duplicated(["sample_date", "instrument", "horizon"]).any():
        raise AssertionError("标签生成后出现重复键")
    return (
        labels.sort_values(["horizon", "sample_date", "instrument"]).reset_index(drop=True),
        coverage_audit.sort_values(["horizon", "sample_date"]).reset_index(drop=True),
    )


def split_design_samples(
    dataset: pd.DataFrame,
    *,
    train_start,
    train_end,
    validation_start,
    validation_end,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split samples and enforce label-complete purge at both cutoffs."""
    required = {"sample_date", "label_end_timestamp"}
    if required - set(dataset.columns):
        raise ValueError(f"样本缺少字段：{sorted(required - set(dataset.columns))}")
    frame = dataset.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"]).dt.normalize()
    frame["label_end_timestamp"] = pd.to_datetime(
        frame["label_end_timestamp"]
    ).dt.normalize()
    train_start = pd.Timestamp(train_start).normalize()
    train_end = pd.Timestamp(train_end).normalize()
    validation_start = pd.Timestamp(validation_start).normalize()
    validation_end = pd.Timestamp(validation_end).normalize()
    if not train_start <= train_end < validation_start <= validation_end:
        raise ValueError("训练/验证窗口顺序无效")

    train = frame.loc[
        frame["sample_date"].between(train_start, train_end)
        & frame["label_end_timestamp"].le(train_end)
    ].copy()
    validation = frame.loc[
        frame["sample_date"].between(validation_start, validation_end)
        & frame["label_end_timestamp"].le(validation_end)
    ].copy()
    if train.empty or validation.empty:
        year_counts = (
            frame.groupby(frame["sample_date"].dt.year).size().astype(int).to_dict()
        )
        raise ValueError(
            "训练集或验证集为空："
            f"train_rows={len(train)}, validation_rows={len(validation)}, "
            f"输入样本年度分布={year_counts}"
        )
    if train["label_end_timestamp"].max() > train_end:
        raise AssertionError("训练集存在跨过训练截止日的标签")
    if validation["label_end_timestamp"].max() > validation_end:
        raise AssertionError("验证集存在跨过验证截止日的标签")
    if train["sample_date"].max() >= validation["sample_date"].min():
        raise AssertionError("训练集与验证集时间重叠")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def cross_sectional_zscore(
    frame: pd.DataFrame, value_col: str, date_col: str = "sample_date"
) -> pd.Series:
    """Match Qlib Alpha158's default CSZScoreNorm label processor."""
    if value_col not in frame or date_col not in frame:
        raise ValueError("截面排序字段不存在")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    means = values.groupby(frame[date_col]).transform("mean")
    std = values.groupby(frame[date_col]).transform(lambda x: x.std(ddof=1))
    result = (values - means) / std.where(std.gt(0))
    return result.astype(float)


def rank_ic_series(
    frame: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    label_col: str = "excess_return",
    date_col: str = "sample_date",
    min_cross_section_size: int = 30,
) -> pd.DataFrame:
    required = {date_col, prediction_col, label_col}
    if required - set(frame.columns):
        raise ValueError(f"IC 输入缺少字段：{sorted(required - set(frame.columns))}")
    rows: list[dict] = []
    for date, group in frame.groupby(date_col, sort=True):
        pair = group[[prediction_col, label_col]].apply(pd.to_numeric, errors="coerce")
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
        if (
            len(pair) < int(min_cross_section_size)
            or pair[prediction_col].nunique() < 2
            or pair[label_col].nunique() < 2
        ):
            ic = np.nan
        else:
            ic = pair[prediction_col].rank(method="average").corr(
                pair[label_col].rank(method="average")
            )
        rows.append(
            {"sample_date": pd.Timestamp(date).normalize(), "rank_ic": ic, "count": len(pair)}
        )
    return pd.DataFrame(rows).sort_values("sample_date").reset_index(drop=True)


def summarize_ic(ic: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("没有可计算的 Rank IC")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    icir = float(mean / std) if np.isfinite(std) and std > 0 else np.nan
    return {"observations": int(len(clean)), "rank_ic_mean": mean, "rank_ic_std": std, "icir": icir}


def icir_by_period(ic_frame: pd.DataFrame) -> pd.DataFrame:
    if {"sample_date", "rank_ic"} - set(ic_frame.columns):
        raise ValueError("ICIR 输入字段不完整")
    frame = ic_frame.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"])
    rows = []
    for year, group in frame.groupby(frame["sample_date"].dt.year):
        rows.append({"period": str(int(year)), **summarize_ic(group["rank_ic"])})
    rows.append({"period": "Full", **summarize_ic(frame["rank_ic"])})
    return pd.DataFrame(rows)


def compute_ic_decay(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: Iterable[int],
    *,
    min_cross_section_size: int = 30,
) -> pd.DataFrame:
    required_predictions = {"sample_date", "instrument", "prediction"}
    required_labels = {"sample_date", "instrument", "horizon", "excess_return"}
    if required_predictions - set(predictions.columns):
        raise ValueError("预测结果字段不完整")
    if required_labels - set(labels.columns):
        raise ValueError("衰减标签字段不完整")
    rows = []
    for horizon in sorted({int(item) for item in horizons}):
        horizon_labels = labels.loc[labels["horizon"].eq(horizon)]
        merged = predictions[list(required_predictions)].merge(
            horizon_labels[["sample_date", "instrument", "excess_return"]],
            on=["sample_date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
        series = rank_ic_series(
            merged, min_cross_section_size=min_cross_section_size
        )
        summary = summarize_ic(series["rank_ic"])
        rows.append({"horizon": horizon, "pair_rows": len(merged), **summary})
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def decile_return_table(
    validation: pd.DataFrame,
    *,
    deciles: int = 10,
    prediction_col: str = "prediction",
    label_col: str = "excess_return",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    required = {"sample_date", "instrument", prediction_col, label_col}
    if required - set(validation.columns):
        raise ValueError("十分组输入字段不完整")
    if int(deciles) < 2:
        raise ValueError("分组数必须至少为 2")
    frame = validation.sort_values(["sample_date", "instrument"]).copy()

    def assign(group: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(group, errors="coerce")
        valid = numeric.notna()
        result = pd.Series(np.nan, index=group.index, dtype=float)
        if valid.any():
            ranks = numeric.loc[valid].rank(method="first").astype(int) - 1
            result.loc[valid] = np.floor(ranks * int(deciles) / valid.sum()) + 1
        return result

    frame["decile"] = frame.groupby("sample_date", group_keys=False)[prediction_col].apply(assign)
    frame = frame.dropna(subset=["decile", label_col]).copy()
    frame["decile"] = frame["decile"].astype(int)
    table = (
        frame.groupby("decile", as_index=False)
        .agg(mean_excess_return=(label_col, "mean"), observations=(label_col, "size"))
        .sort_values("decile")
        .reset_index(drop=True)
    )
    if len(table) != int(deciles):
        raise ValueError(f"有效分组只有 {len(table)} 个，期望 {deciles} 个")
    means = table["mean_excess_return"].to_numpy(dtype=float)
    adjacent_up_steps = int(np.sum(np.diff(means) >= 0))
    monotonic_spearman = float(
        pd.Series(table["decile"], dtype=float).corr(
            pd.Series(means), method="spearman"
        )
    )
    summary = {
        "adjacent_up_steps": adjacent_up_steps,
        "possible_adjacent_steps": int(deciles) - 1,
        "top_minus_bottom": float(means[-1] - means[0]),
        "decile_spearman": monotonic_spearman,
    }
    return table, summary


def evaluate_acceptance(
    ic_summary: dict,
    monotonicity: dict,
    *,
    rank_ic_mean_min: float,
    icir_min: float,
    monotonic_min_adjacent_steps: int,
) -> dict[str, bool]:
    acceptance = {
        "rank_ic_mean": bool(ic_summary["rank_ic_mean"] > rank_ic_mean_min),
        "icir": bool(ic_summary["icir"] > icir_min),
        "decile_monotonicity": bool(
            monotonicity["adjacent_up_steps"] >= monotonic_min_adjacent_steps
            and monotonicity["top_minus_bottom"] > 0
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    return acceptance

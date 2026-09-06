from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtri


class CalibrationError(ValueError):
    """Raised when scores cannot be mapped to an expected-return scale."""


@dataclass(frozen=True)
class ExpandingFold:
    prediction_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        for key in ["train_start", "train_end", "prediction_start", "prediction_end"]:
            record[key] = str(pd.Timestamp(record[key]).date())
        return record


def build_expanding_year_folds(
    *,
    train_start,
    initial_train_end,
    prediction_start,
    prediction_end,
) -> list[ExpandingFold]:
    """Build calendar-year prediction folds with an expanding training window."""
    train_start = pd.Timestamp(train_start).normalize()
    initial_train_end = pd.Timestamp(initial_train_end).normalize()
    prediction_start = pd.Timestamp(prediction_start).normalize()
    prediction_end = pd.Timestamp(prediction_end).normalize()
    if not train_start <= initial_train_end < prediction_start <= prediction_end:
        raise ValueError("扩展窗口日期顺序无效")
    if initial_train_end != prediction_start - pd.Timedelta(days=1):
        raise ValueError("首折训练截止日必须紧邻首个预测区间")

    folds: list[ExpandingFold] = []
    for year in range(prediction_start.year, prediction_end.year + 1):
        fold_start = max(prediction_start, pd.Timestamp(year=year, month=1, day=1))
        fold_end = min(prediction_end, pd.Timestamp(year=year, month=12, day=31))
        if fold_start > fold_end:
            continue
        folds.append(
            ExpandingFold(
                prediction_year=year,
                train_start=train_start,
                train_end=fold_start - pd.Timedelta(days=1),
                prediction_start=fold_start,
                prediction_end=fold_end,
            )
        )
    if not folds:
        raise ValueError("没有生成任何扩展窗口折")
    return folds


def split_walk_forward_samples(
    dataset: pd.DataFrame, fold: ExpandingFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one labeled fold and purge labels crossing either evaluation cutoff."""
    required = {"sample_date", "label_end_timestamp"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"滚动样本缺少字段：{sorted(missing)}")
    frame = dataset.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"]).dt.normalize()
    frame["label_end_timestamp"] = pd.to_datetime(
        frame["label_end_timestamp"]
    ).dt.normalize()
    train = frame.loc[
        frame["sample_date"].between(fold.train_start, fold.train_end)
        & frame["label_end_timestamp"].le(fold.train_end)
    ].copy()
    prediction = frame.loc[
        frame["sample_date"].between(fold.prediction_start, fold.prediction_end)
        & frame["label_end_timestamp"].le(fold.prediction_end)
    ].copy()
    assert_fold_integrity(train, prediction, fold)
    return train.reset_index(drop=True), prediction.reset_index(drop=True)


def assert_fold_integrity(
    train: pd.DataFrame, prediction: pd.DataFrame, fold: ExpandingFold
) -> dict[str, object]:
    """Enforce purge and chronology for one walk-forward fold."""
    if train.empty or prediction.empty:
        raise ValueError(f"{fold.prediction_year} 折训练集或预测集为空")
    train_sample = pd.to_datetime(train["sample_date"]).dt.normalize()
    train_label_end = pd.to_datetime(train["label_end_timestamp"]).dt.normalize()
    prediction_sample = pd.to_datetime(prediction["sample_date"]).dt.normalize()
    prediction_label_end = pd.to_datetime(
        prediction["label_end_timestamp"]
    ).dt.normalize()
    if train_sample.min() < fold.train_start or train_sample.max() > fold.train_end:
        raise AssertionError("训练样本日期越过当前折窗口")
    if train_label_end.max() > fold.train_end:
        raise AssertionError("训练集最后一个标签结束日越过当前折训练截止日")
    if (
        prediction_sample.min() < fold.prediction_start
        or prediction_sample.max() > fold.prediction_end
    ):
        raise AssertionError("预测样本日期越过当前折窗口")
    if prediction_label_end.max() > fold.prediction_end:
        raise AssertionError("设计期预测标签越过当前折评价截止日")
    if train_sample.max() >= prediction_sample.min():
        raise AssertionError("训练样本与预测样本发生时间重叠")
    return {
        **fold.as_record(),
        "train_rows": int(len(train)),
        "train_dates": int(train_sample.nunique()),
        "prediction_rows": int(len(prediction)),
        "prediction_dates": int(prediction_sample.nunique()),
        "last_train_sample_date": str(train_sample.max().date()),
        "last_train_label_end": str(train_label_end.max().date()),
        "first_prediction_sample_date": str(prediction_sample.min().date()),
        "last_prediction_sample_date": str(prediction_sample.max().date()),
        "purge_passed": True,
    }


def rank_gaussianize(
    frame: pd.DataFrame,
    *,
    score_col: str = "prediction",
    date_col: str = "sample_date",
) -> pd.Series:
    """Convert each cross-section's score ranks to finite standard-normal scores."""
    required = {score_col, date_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"分数正态化缺少字段：{sorted(missing)}")
    values = pd.to_numeric(frame[score_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    def transform(group: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=group.index, dtype=float)
        valid = group.dropna()
        if len(valid) < 2 or valid.nunique() < 2:
            return result
        ranks = valid.rank(method="average")
        probabilities = (ranks - 0.5) / len(valid)
        normal = pd.Series(ndtri(probabilities.to_numpy()), index=valid.index)
        std = float(normal.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            return result
        result.loc[valid.index] = (normal - normal.mean()) / std
        return result

    normalized = values.groupby(frame[date_col], group_keys=False).apply(transform)
    if isinstance(normalized.index, pd.MultiIndex):
        normalized = normalized.droplevel(0)
    return normalized.reindex(frame.index).astype(float)


def fit_return_calibration(
    frame: pd.DataFrame,
    *,
    score_col: str = "score_z",
    label_col: str = "excess_return",
    date_col: str = "sample_date",
) -> dict[str, object]:
    """Fit y = a*z with equal total weight for every weekly cross-section."""
    required = {score_col, label_col, date_col}
    missing = required - set(frame.columns)
    if missing:
        raise CalibrationError(f"校准输入缺少字段：{sorted(missing)}")
    data = frame[list(required)].copy()
    data[date_col] = pd.to_datetime(data[date_col]).dt.normalize()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")
    finite = np.isfinite(data[score_col]) & np.isfinite(data[label_col])
    data = data.loc[finite].copy()
    if data.empty or data[date_col].nunique() < 1:
        raise CalibrationError("没有可用于收益量纲校准的样本")
    counts = data.groupby(date_col)[score_col].transform("size")
    weights = 1.0 / counts.astype(float)
    denominator = float((weights * data[score_col].pow(2)).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise CalibrationError("校准分数没有有效截面方差")
    numerator = float((weights * data[score_col] * data[label_col]).sum())
    coefficient = numerator / denominator
    if not np.isfinite(coefficient):
        raise CalibrationError("校准系数不是有限数")
    correlation = float(data[score_col].corr(data[label_col]))
    return {
        "method": "date_balanced_zero_intercept_ols",
        "coefficient": float(coefficient),
        "observations": int(len(data)),
        "dates": int(data[date_col].nunique()),
        "history_start": str(data[date_col].min().date()),
        "history_end": str(data[date_col].max().date()),
        "score_label_pearson": correlation,
    }


def apply_return_calibration(score_z: pd.Series, coefficient: float) -> pd.Series:
    coefficient = float(coefficient)
    if not np.isfinite(coefficient):
        raise CalibrationError("校准系数必须是有限数")
    score = pd.to_numeric(score_z, errors="coerce")
    return (score * coefficient).astype(float)


def realized_alpha_deciles(
    frame: pd.DataFrame,
    *,
    deciles: int,
    score_col: str = "score_z",
    label_col: str = "excess_return",
    date_col: str = "sample_date",
) -> pd.DataFrame:
    """Estimate the realized alpha curve from validation score deciles."""
    if int(deciles) < 2:
        raise ValueError("校准分组数必须至少为 2")
    required = {score_col, label_col, date_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"已实现 alpha 尺度输入缺少字段：{sorted(missing)}")
    data = frame.copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")

    def assign(group: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=group.index, dtype=float)
        valid = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < int(deciles):
            return result
        rank = valid.rank(method="first").astype(int) - 1
        result.loc[valid.index] = np.floor(rank * int(deciles) / len(valid)) + 1
        return result

    data["calibration_decile"] = data.groupby(
        date_col, group_keys=False
    )[score_col].apply(assign)
    data = data.dropna(subset=["calibration_decile", label_col]).copy()
    data["calibration_decile"] = data["calibration_decile"].astype(int)
    table = (
        data.groupby("calibration_decile", as_index=False)
        .agg(
            mean_realized_excess=(label_col, "mean"),
            observations=(label_col, "size"),
        )
        .sort_values("calibration_decile")
        .reset_index(drop=True)
    )
    if len(table) != int(deciles):
        raise CalibrationError(
            f"已实现 alpha 曲线只有 {len(table)} 个分组，期望 {int(deciles)} 个"
        )
    return table


def calibration_scale_diagnostics(
    frame: pd.DataFrame,
    coefficient: float,
    *,
    deciles: int,
    tolerance: float,
    score_col: str = "score_z",
    label_col: str = "excess_return",
    date_col: str = "sample_date",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare calibrated mu dispersion with the validation realized-alpha curve."""
    tolerance = float(tolerance)
    if not 0 <= tolerance < 1:
        raise ValueError("校准尺度容忍度必须位于 [0, 1)")
    data = frame.copy()
    data["expected_5d_excess"] = apply_return_calibration(
        data[score_col], coefficient
    )
    mu_std = (
        data.groupby(date_col)["expected_5d_excess"]
        .std(ddof=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if mu_std.empty:
        raise CalibrationError("没有可计算的校准后截面标准差")
    curve = realized_alpha_deciles(
        data,
        deciles=int(deciles),
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
    )
    realized_scale = float(curve["mean_realized_excess"].std(ddof=1))
    calibrated_scale = float(mu_std.mean())
    if not np.isfinite(realized_scale) or realized_scale <= 0:
        raise CalibrationError("验证集已实现 alpha 尺度无效")
    lower = realized_scale * (1.0 - tolerance)
    upper = realized_scale * (1.0 + tolerance)
    return (
        {
            "calibrated_mu_mean_cross_section_std": calibrated_scale,
            "realized_alpha_decile_std": realized_scale,
            "scale_ratio": calibrated_scale / realized_scale,
            "tolerance": tolerance,
            "lower_bound": lower,
            "upper_bound": upper,
            "passed": bool(lower <= calibrated_scale <= upper),
            "realized_alpha_scale_method": (
                "std of validation decile mean realized 5-day excess returns"
            ),
        },
        curve,
    )


def build_annual_calibration_schedule(
    predictions: pd.DataFrame,
    effective_years: Iterable[int],
    *,
    minimum_history_years: int = 1,
    score_col: str = "score_z",
    label_col: str = "excess_return",
    date_col: str = "sample_date",
) -> pd.DataFrame:
    """Refit calibration using only realized walk-forward predictions from prior years."""
    frame = predictions.copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    rows: list[dict[str, object]] = []
    for year in sorted({int(item) for item in effective_years}):
        effective_start = pd.Timestamp(year=year, month=1, day=1)
        history = frame.loc[frame[date_col].lt(effective_start)].copy()
        history_years = sorted(history[date_col].dt.year.unique().tolist())
        if len(history_years) < int(minimum_history_years):
            raise CalibrationError(
                f"{year} 年校准历史只有 {len(history_years)} 年，"
                f"要求至少 {int(minimum_history_years)} 年"
            )
        result = fit_return_calibration(
            history,
            score_col=score_col,
            label_col=label_col,
            date_col=date_col,
        )
        if pd.Timestamp(result["history_end"]) >= effective_start:
            raise AssertionError("校准历史越过系数生效年份")
        rows.append(
            {
                "effective_prediction_year": year,
                "calibration_coefficient": result["coefficient"],
                "history_start": result["history_start"],
                "history_end": result["history_end"],
                "history_years": len(history_years),
                "observations": result["observations"],
                "dates": result["dates"],
                "history_precedes_effective_year": True,
            }
        )
    if not rows:
        raise CalibrationError("没有生成年度校准计划")
    return pd.DataFrame(rows)

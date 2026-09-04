from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


STYLE_FACTORS = (
    "Size",
    "NonlinearSize",
    "Beta",
    "Momentum",
    "ResidualVolatility",
    "Liquidity",
)


class FactorExposureError(ValueError):
    """Raised when the D6 exposure contract cannot be satisfied."""


@dataclass(frozen=True)
class CovarianceEstimate:
    as_of_date: pd.Timestamp
    instruments: tuple[str, ...]
    covariance: np.ndarray
    shrinkage: float
    observations: int
    excluded_instruments: tuple[str, ...]


def tushare_to_qlib(ts_code: str) -> str:
    """Convert a mainland Tushare stock code to Qlib's exchange-first form."""
    value = str(ts_code).strip().upper()
    match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", value)
    if not match:
        raise FactorExposureError(f"无法转换 Tushare 股票代码：{ts_code!r}")
    number, exchange = match.groups()
    return f"{exchange}{number}"


def qlib_to_tushare(instrument: str) -> str:
    value = str(instrument).strip().upper()
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", value)
    if not match:
        raise FactorExposureError(f"无法转换 Qlib 股票代码：{instrument!r}")
    exchange, number = match.groups()
    return f"{number}.{exchange}"


def industry_factor_name(l1_code: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", str(l1_code).strip()).strip("_")
    if not clean:
        raise FactorExposureError(f"申万一级代码无效：{l1_code!r}")
    return f"industry_{clean}"


def resolve_static_industry_mapping(
    members: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    expected_industry_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve one deliberately non-PIT SW L1 assignment for every stock.

    Current records (``is_new=Y``) win.  For stocks no longer present in the
    current classification, the most recent historical membership wins.  The
    result intentionally drops membership dates: it is a fixed taxonomy, not a
    point-in-time history.
    """
    member_required = {"l1_code", "ts_code"}
    catalog_required = {"l1_code", "industry_name"}
    if member_required - set(members.columns):
        raise FactorExposureError(
            f"行业成员表缺少字段：{sorted(member_required - set(members.columns))}"
        )
    if catalog_required - set(catalog.columns):
        raise FactorExposureError(
            f"行业目录缺少字段：{sorted(catalog_required - set(catalog.columns))}"
        )

    cat = catalog[list(catalog_required)].copy()
    cat["l1_code"] = cat["l1_code"].astype(str).str.strip().str.upper()
    cat["industry_name"] = cat["industry_name"].astype(str).str.strip()
    if cat["l1_code"].eq("").any() or cat["industry_name"].eq("").any():
        raise FactorExposureError("行业目录包含空代码或空名称")
    if cat["l1_code"].duplicated().any():
        raise FactorExposureError("行业目录的申万一级代码不唯一")
    if expected_industry_count is not None and len(cat) != int(expected_industry_count):
        raise FactorExposureError(
            f"申万一级行业数量为 {len(cat)}，期望 {int(expected_industry_count)}"
        )
    cat["industry_factor"] = cat["l1_code"].map(industry_factor_name)
    if cat["industry_factor"].duplicated().any():
        raise FactorExposureError("清洗后的行业因子名不唯一")

    work = members.copy()
    work["l1_code"] = work["l1_code"].astype(str).str.strip().str.upper()
    work["ts_code"] = work["ts_code"].astype(str).str.strip().str.upper()
    mainland_stock = work["ts_code"].str.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)")
    work = work.loc[
        work["l1_code"].isin(cat["l1_code"]) & mainland_stock.fillna(False)
    ].copy()
    if work.empty:
        raise FactorExposureError("行业成员表与 SW2021 一级目录没有交集")
    work["current_priority"] = (
        work.get("is_new", pd.Series("N", index=work.index))
        .astype(str)
        .str.upper()
        .eq("Y")
        .astype(int)
    )
    in_date = pd.to_datetime(
        work["in_date"] if "in_date" in work else pd.Series(pd.NaT, index=work.index),
        errors="coerce",
    )
    out_date = pd.to_datetime(
        work["out_date"] if "out_date" in work else pd.Series(pd.NaT, index=work.index),
        errors="coerce",
    )
    work["recency"] = out_date.fillna(in_date).fillna(pd.Timestamp("1900-01-01"))

    # Conflicting current assignments are data errors; choosing one would make
    # the fixed taxonomy depend on API row order.
    active = work.loc[work["current_priority"].eq(1)]
    conflicts = active.groupby("ts_code")["l1_code"].nunique()
    conflicts = conflicts.loc[conflicts.gt(1)]
    if not conflicts.empty:
        raise FactorExposureError(
            "同一股票存在多个当前申万一级归属：" + ", ".join(conflicts.index[:10])
        )

    chosen = (
        work.sort_values(
            ["ts_code", "current_priority", "recency", "l1_code"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("ts_code", keep="first")
        .loc[:, ["ts_code", "l1_code"]]
        .merge(cat, on="l1_code", how="left", validate="many_to_one")
    )
    chosen["instrument"] = chosen["ts_code"].map(tushare_to_qlib)
    mapping = chosen[
        ["instrument", "ts_code", "l1_code", "industry_name", "industry_factor"]
    ].sort_values("instrument").reset_index(drop=True)
    if mapping["instrument"].duplicated().any():
        raise FactorExposureError("固定行业映射存在重复股票")
    return mapping, cat.sort_values("l1_code").reset_index(drop=True)


def require_industry_coverage(
    instruments: Iterable[str], mapping: pd.DataFrame
) -> pd.DataFrame:
    required = {"instrument", "l1_code", "industry_factor"}
    if required - set(mapping.columns):
        raise FactorExposureError(
            f"固定行业映射缺少字段：{sorted(required - set(mapping.columns))}"
        )
    requested = pd.Index(sorted({str(item).upper() for item in instruments}))
    available = pd.Index(mapping["instrument"].astype(str).str.upper().unique())
    missing = requested.difference(available)
    if len(missing):
        preview = ", ".join(missing[:20])
        raise FactorExposureError(
            f"固定申万一级映射缺少 {len(missing)} 只设计期成分股：{preview}"
        )
    return mapping.loc[mapping["instrument"].isin(requested)].copy()


def winsorize_mad(values: pd.Series, multiple: float = 3.0) -> pd.Series:
    """Median-MAD winsorization with the normal-consistency scale."""
    if float(multiple) <= 0:
        raise FactorExposureError("MAD 去极值倍数必须为正")
    result = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = result.dropna()
    if valid.empty:
        return result.astype(float)
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        return result.astype(float)
    width = float(multiple) * 1.4826 * mad
    return result.clip(lower=median - width, upper=median + width).astype(float)


def _weighted_residual(
    dependent: pd.Series, independent: pd.Series, weights: pd.Series
) -> pd.Series:
    y = pd.to_numeric(dependent, errors="coerce")
    x = pd.to_numeric(independent, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    valid = y.notna() & x.notna() & w.gt(0) & np.isfinite(w)
    if valid.sum() < 3 or x.loc[valid].nunique() < 2:
        raise FactorExposureError("Non-linear Size 无法对 Size 做加权正交化")
    design = np.column_stack([np.ones(valid.sum()), x.loc[valid].to_numpy(float)])
    root_w = np.sqrt(w.loc[valid].to_numpy(float))
    coefficients, *_ = np.linalg.lstsq(
        design * root_w[:, None], y.loc[valid].to_numpy(float) * root_w, rcond=None
    )
    result = pd.Series(np.nan, index=y.index, dtype=float)
    result.loc[valid] = y.loc[valid] - design @ coefficients
    return result


def normalize_style(
    values: pd.Series,
    market_cap: pd.Series,
    *,
    winsor_mad_multiple: float,
    apply_winsor: bool = True,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Winsorize, market-cap demean, neutral-impute, then EW-scale."""
    raw = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cap = pd.to_numeric(market_cap, errors="coerce")
    treated = winsorize_mad(raw, winsor_mad_multiple) if apply_winsor else raw
    valid = treated.notna() & cap.gt(0) & np.isfinite(cap)
    if valid.sum() < 2 or treated.loc[valid].nunique() < 2:
        raise FactorExposureError("风格因子有效样本或截面方差不足")
    weighted_mean = float(np.average(treated.loc[valid], weights=cap.loc[valid]))
    centered = treated - weighted_mean
    centered.loc[~valid] = 0.0
    scale = float(centered.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise FactorExposureError("风格因子等权标准差为零")
    normalized = centered / scale
    normalized_weighted_mean = float(np.average(normalized, weights=cap.where(cap.gt(0), 0.0)))
    diagnostics: dict[str, float | int] = {
        "rows": int(len(raw)),
        "valid_raw": int(valid.sum()),
        "imputed_neutral": int((~valid).sum()),
        "raw_coverage": float(valid.mean()),
        "market_cap_weighted_mean": normalized_weighted_mean,
        "equal_weight_mean": float(normalized.mean()),
        "equal_weight_std": float(normalized.std(ddof=0)),
    }
    return normalized.astype(float), diagnostics


def process_style_cross_section(
    frame: pd.DataFrame,
    *,
    winsor_mad_multiple: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the six D6 style exposures for one date."""
    required = {
        "instrument",
        "market_cap",
        "raw_Size",
        "raw_Beta",
        "raw_Momentum",
        "raw_ResidualVolatility",
        "raw_Liquidity",
    }
    if required - set(frame.columns):
        raise FactorExposureError(
            f"风格截面缺少字段：{sorted(required - set(frame.columns))}"
        )
    if frame["instrument"].duplicated().any():
        raise FactorExposureError("风格截面存在重复股票")
    result = frame.copy()
    cap = pd.to_numeric(result["market_cap"], errors="coerce")
    if cap.isna().any() or cap.le(0).any():
        raise FactorExposureError("市值加权要求每只股票都有正总市值")

    size_winsor = winsorize_mad(result["raw_Size"], winsor_mad_multiple)
    size, size_diag = normalize_style(
        size_winsor,
        cap,
        winsor_mad_multiple=winsor_mad_multiple,
        apply_winsor=False,
    )
    result["style_Size"] = size

    cube = winsorize_mad(size_winsor.pow(3), winsor_mad_multiple)
    nonlinear_residual = _weighted_residual(cube, size_winsor, cap)
    nonlinear, nonlinear_diag = normalize_style(
        nonlinear_residual,
        cap,
        winsor_mad_multiple=winsor_mad_multiple,
        apply_winsor=False,
    )
    result["style_NonlinearSize"] = nonlinear

    diagnostics = [("Size", size_diag), ("NonlinearSize", nonlinear_diag)]
    for factor in ["Beta", "Momentum", "ResidualVolatility", "Liquidity"]:
        normalized, diag = normalize_style(
            result[f"raw_{factor}"],
            cap,
            winsor_mad_multiple=winsor_mad_multiple,
        )
        result[f"style_{factor}"] = normalized
        diagnostics.append((factor, diag))

    audit = pd.DataFrame(
        [{"style_factor": factor, **values} for factor, values in diagnostics]
    )
    return result, audit


def compute_raw_style_history(
    close_prices: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    daily_basic: pd.DataFrame,
    *,
    beta_window: int = 252,
    beta_minimum_observations: int = 126,
    momentum_window: int = 252,
    momentum_skip: int = 21,
    liquidity_window: int = 21,
    liquidity_minimum_observations: int = 5,
) -> pd.DataFrame:
    """Compute causal raw style descriptors on a daily panel."""
    price_required = {"trade_date", "instrument", "close"}
    benchmark_required = {"trade_date", "benchmark_return"}
    basic_required = {"trade_date", "instrument", "total_mv", "turnover_rate"}
    for name, value, required in [
        ("收盘价", close_prices, price_required),
        ("基准收益", benchmark_returns, benchmark_required),
        ("daily_basic", daily_basic, basic_required),
    ]:
        missing = required - set(value.columns)
        if missing:
            raise FactorExposureError(f"{name}缺少字段：{sorted(missing)}")
    if not 0 < int(momentum_skip) < int(momentum_window):
        raise FactorExposureError("Momentum 必须满足 0 < skip < window")

    prices = close_prices[list(price_required)].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if prices.duplicated(["trade_date", "instrument"]).any():
        raise FactorExposureError("收盘价存在重复的 日期×股票")
    close = prices.pivot(index="trade_date", columns="instrument", values="close").sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    bench = benchmark_returns[list(benchmark_required)].copy()
    bench["trade_date"] = pd.to_datetime(bench["trade_date"]).dt.normalize()
    bench["benchmark_return"] = pd.to_numeric(bench["benchmark_return"], errors="coerce")
    if bench["trade_date"].duplicated().any():
        raise FactorExposureError("基准收益存在重复日期")
    benchmark = bench.set_index("trade_date")["benchmark_return"].reindex(close.index)

    rolling_stock_var = returns.rolling(
        int(beta_window), min_periods=int(beta_minimum_observations)
    ).var()
    rolling_cov = returns.rolling(
        int(beta_window), min_periods=int(beta_minimum_observations)
    ).cov(benchmark)
    rolling_benchmark_var = benchmark.rolling(
        int(beta_window), min_periods=int(beta_minimum_observations)
    ).var()
    beta = rolling_cov.div(rolling_benchmark_var, axis=0)
    residual_variance = rolling_stock_var - rolling_cov.pow(2).div(
        rolling_benchmark_var, axis=0
    )
    residual_volatility = np.sqrt(residual_variance.clip(lower=0.0))
    momentum = close.shift(int(momentum_skip)).div(close.shift(int(momentum_window))) - 1.0

    basic = daily_basic[list(basic_required)].copy()
    basic["trade_date"] = pd.to_datetime(basic["trade_date"]).dt.normalize()
    for column in ["total_mv", "turnover_rate"]:
        basic[column] = pd.to_numeric(basic[column], errors="coerce")
    if basic.duplicated(["trade_date", "instrument"]).any():
        raise FactorExposureError("daily_basic 存在重复的 日期×股票")
    market_cap = basic.pivot(
        index="trade_date", columns="instrument", values="total_mv"
    ).reindex(index=close.index, columns=close.columns).ffill()
    turnover = basic.pivot(
        index="trade_date", columns="instrument", values="turnover_rate"
    ).reindex(index=close.index, columns=close.columns)
    liquidity = np.log(
        turnover.rolling(
            int(liquidity_window), min_periods=int(liquidity_minimum_observations)
        ).mean()
    )

    matrices = {
        "market_cap": market_cap,
        "raw_Size": np.log(market_cap.where(market_cap.gt(0))),
        "raw_Beta": beta,
        "raw_Momentum": momentum,
        "raw_ResidualVolatility": residual_volatility,
        "raw_Liquidity": liquidity,
    }
    pieces = []
    for name, matrix in matrices.items():
        series = matrix.stack(future_stack=True)
        series.name = name
        pieces.append(series)
    result = pd.concat(pieces, axis=1).reset_index()
    result.columns = ["trade_date", "instrument", *matrices.keys()]
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def build_industry_dummies(
    frame: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    required = {"industry_factor"}
    if required - set(frame.columns) or required - set(catalog.columns):
        raise FactorExposureError("行业哑变量输入缺少 industry_factor")
    columns = catalog.sort_values("l1_code")["industry_factor"].tolist()
    if len(columns) != len(set(columns)):
        raise FactorExposureError("行业哑变量列名重复")
    dummies = pd.get_dummies(frame["industry_factor"], dtype=float)
    unknown = sorted(set(dummies.columns) - set(columns))
    if unknown:
        raise FactorExposureError(f"截面出现目录外行业：{unknown}")
    dummies = dummies.reindex(columns=columns, fill_value=0.0)
    result = pd.concat([frame.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    row_sum = result[columns].sum(axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=0.0):
        raise FactorExposureError("行业哑变量每行之和不等于 1")
    return result, columns


def audit_factor_matrix(
    frame: pd.DataFrame,
    industry_columns: list[str],
    style_columns: list[str],
    *,
    normalization_tolerance: float = 1e-10,
    industry_exposure_tolerance: float = 1e-12,
) -> dict[str, float | int | bool]:
    required = {
        "instrument",
        "market_cap",
        "benchmark_weight",
        "industry_factor",
        *industry_columns,
        *style_columns,
    }
    if required - set(frame.columns):
        raise FactorExposureError(
            f"暴露矩阵审计缺少字段：{sorted(required - set(frame.columns))}"
        )
    global_matrix = frame[industry_columns + style_columns].to_numpy(float)
    if not np.isfinite(global_matrix).all():
        raise FactorExposureError("暴露矩阵包含非有限数")
    # A fixed 31-industry schema is kept in the stored panel.  On a particular
    # date, however, an industry with no benchmark constituent is an all-zero
    # column, not an estimable factor.  Rank is therefore defined on that day's
    # active industry columns plus all six style columns.
    active_industry_columns = [
        column
        for column in industry_columns
        if bool(frame[column].abs().gt(float(industry_exposure_tolerance)).any())
    ]
    inactive_industry_columns = sorted(set(industry_columns) - set(active_industry_columns))
    effective_columns = active_industry_columns + style_columns
    effective_matrix = frame[effective_columns].to_numpy(float)
    rank = int(np.linalg.matrix_rank(effective_matrix))
    global_rank = int(np.linalg.matrix_rank(global_matrix))
    full_rank = rank == effective_matrix.shape[1]
    industry_row_error = float(np.abs(frame[industry_columns].sum(axis=1) - 1.0).max())

    cap = pd.to_numeric(frame["market_cap"], errors="coerce")
    cap_weight = cap / cap.sum()
    weighted_means = frame[style_columns].mul(cap_weight, axis=0).sum(axis=0)
    style_stds = frame[style_columns].std(axis=0, ddof=0)
    max_weighted_mean = float(weighted_means.abs().max())
    max_std_error = float((style_stds - 1.0).abs().max())

    benchmark_weight = pd.to_numeric(frame["benchmark_weight"], errors="coerce")
    if benchmark_weight.isna().any() or benchmark_weight.lt(0).any():
        raise FactorExposureError("基准权重包含缺失或负数")
    benchmark_weight = benchmark_weight / benchmark_weight.sum()
    via_matrix = frame[industry_columns].mul(benchmark_weight, axis=0).sum(axis=0)
    direct = benchmark_weight.groupby(frame["industry_factor"]).sum().reindex(
        industry_columns, fill_value=0.0
    )
    industry_exposure_error = float((via_matrix - direct).abs().max())

    passed = bool(
        full_rank
        and industry_row_error <= float(industry_exposure_tolerance)
        and max_weighted_mean <= float(normalization_tolerance)
        and max_std_error <= float(normalization_tolerance)
        and industry_exposure_error <= float(industry_exposure_tolerance)
    )
    return {
        "rows": int(len(frame)),
        "columns": int(effective_matrix.shape[1]),
        "global_columns": int(global_matrix.shape[1]),
        "active_industry_columns": int(len(active_industry_columns)),
        "inactive_industry_columns": ",".join(inactive_industry_columns),
        "matrix_rank": rank,
        "global_matrix_rank": global_rank,
        "full_rank": full_rank,
        "industry_row_sum_max_error": industry_row_error,
        "style_market_cap_weighted_mean_max_abs": max_weighted_mean,
        "style_equal_weight_std_max_error": max_std_error,
        "benchmark_industry_exposure_max_error": industry_exposure_error,
        "passed": passed,
    }


def benchmark_factor_exposure(
    frame: pd.DataFrame, factor_columns: list[str]
) -> pd.Series:
    weights = pd.to_numeric(frame["benchmark_weight"], errors="coerce")
    if weights.isna().any() or weights.lt(0).any() or weights.sum() <= 0:
        raise FactorExposureError("无法计算指数因子暴露：基准权重无效")
    weights = weights / weights.sum()
    return frame[factor_columns].mul(weights, axis=0).sum(axis=0)


def estimate_ledoit_wolf_covariance(
    stock_returns: pd.DataFrame,
    *,
    as_of_date,
    instruments: Iterable[str],
    window: int = 252,
    minimum_observations: int = 126,
) -> CovarianceEstimate:
    required = {"trade_date", "instrument", "return"}
    if required - set(stock_returns.columns):
        raise FactorExposureError(
            f"协方差收益输入缺少字段：{sorted(required - set(stock_returns.columns))}"
        )
    as_of = pd.Timestamp(as_of_date).normalize()
    requested = tuple(sorted({str(item) for item in instruments}))
    work = stock_returns[list(required)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.normalize()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.loc[work["trade_date"].le(as_of) & work["instrument"].isin(requested)]
    pivot = work.pivot(index="trade_date", columns="instrument", values="return").sort_index()
    pivot = pivot.tail(int(window)).reindex(columns=requested)
    counts = pivot.notna().sum()
    eligible = counts.loc[counts.ge(int(minimum_observations))].index.tolist()
    excluded = tuple(sorted(set(requested) - set(eligible)))
    if len(eligible) < 2 or len(pivot) < 2:
        raise FactorExposureError("Ledoit-Wolf 协方差的有效股票或观察数不足")
    sample = pivot[eligible].copy()
    sample = sample.fillna(sample.mean(axis=0))
    if sample.isna().any().any():
        raise FactorExposureError("Ledoit-Wolf 协方差均值填补后仍有缺失")
    estimator = LedoitWolf(assume_centered=False).fit(sample.to_numpy(float))
    covariance = np.asarray(estimator.covariance_, dtype=float)
    if covariance.shape != (len(eligible), len(eligible)):
        raise FactorExposureError("Ledoit-Wolf 协方差维度异常")
    if np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise FactorExposureError("Ledoit-Wolf 协方差不是半正定矩阵")
    return CovarianceEstimate(
        as_of_date=as_of,
        instruments=tuple(eligible),
        covariance=covariance,
        shrinkage=float(estimator.shrinkage_),
        observations=int(len(sample)),
        excluded_instruments=excluded,
    )

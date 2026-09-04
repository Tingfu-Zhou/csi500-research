from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d4_pipeline import atomic_json, atomic_parquet, load_calendar  # noqa: E402
from src.d4_core import select_weekly_sample_dates  # noqa: E402
from src.d6_core import (  # noqa: E402
    STYLE_FACTORS,
    FactorExposureError,
    audit_factor_matrix,
    benchmark_factor_exposure,
    build_industry_dummies,
    compute_raw_style_history,
    estimate_ledoit_wolf_covariance,
    process_style_cross_section,
    require_industry_coverage,
    tushare_to_qlib,
)
from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def require_passed_d5(reports_dir: Path) -> dict:
    path = reports_dir / "d5_summary.json"
    if not path.exists():
        raise FileNotFoundError("缺少 reports/d5_summary.json，请先成功运行 D5")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "design":
        raise RuntimeError("D5 结果不是 design 模式产物")
    if not payload.get("acceptance", {}).get("passed", False):
        raise RuntimeError("D5 未通过写死的验收标准，不得进入 D6")
    return payload


def load_close_prices(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    raw = access.features(instruments, ["$close"], start_date, end_date)
    if raw.empty:
        raise ValueError("Qlib 没有返回 D6 收盘价")
    raw = raw.copy()
    raw.columns = ["close"]
    frame = raw.reset_index().rename(columns={"datetime": "trade_date"})
    frame["trade_date"] = parse_date_series(frame["trade_date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def load_benchmark_returns(
    access: ResearchDataAccess,
    config: dict,
    start_date,
    end_date,
) -> pd.DataFrame:
    path = Path(config["data"]["tushare_raw_dir"]) / "index_daily_H00905_CSI.parquet"
    frame = access.read_parquet(
        path,
        date_col="trade_date",
        start_date=start_date,
        end_date=end_date,
    )
    if "close" not in frame:
        raise ValueError("H00905.CSI 全收益指数文件缺少 close")
    frame["trade_date"] = parse_date_series(frame["trade_date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.drop_duplicates("trade_date").sort_values("trade_date")
    frame["benchmark_return"] = frame["close"].pct_change(fill_method=None)
    return frame[["trade_date", "benchmark_return"]].reset_index(drop=True)


def load_daily_basic(
    access: ResearchDataAccess,
    config: dict,
    instruments: list[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    requested = set(instruments)
    frames = []
    for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
        relative = raw_dir / f"daily_basic_{year}.parquet"
        if not (ROOT / relative).exists():
            continue
        frame = access.read_parquet(
            relative,
            date_col="trade_date",
            start_date=start_date,
            end_date=end_date,
        )
        if "instrument" not in frame:
            if "ts_code" not in frame:
                raise ValueError(f"{relative} 缺少 instrument/ts_code")
            frame["instrument"] = frame["ts_code"].map(tushare_to_qlib)
        frame["instrument"] = frame["instrument"].astype(str).str.upper()
        frames.append(frame.loc[frame["instrument"].isin(requested)].copy())
    if not frames:
        raise FileNotFoundError("没有找到 D6 所需 daily_basic_YYYY.parquet")
    result = pd.concat(frames, ignore_index=True, sort=False)
    required = ["trade_date", "instrument", "total_mv", "turnover_rate"]
    missing = set(required) - set(result.columns)
    if missing:
        raise ValueError(f"daily_basic 缺少字段：{sorted(missing)}")
    result["trade_date"] = parse_date_series(result["trade_date"])
    return result[required].sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def require_industry_files(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_path = ROOT / config["d6"]["industry_mapping_file"]
    catalog_path = ROOT / config["d6"]["industry_catalog_file"]
    missing = [str(path.relative_to(ROOT)) for path in [mapping_path, catalog_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少固定 SW2021 行业文件："
            + ", ".join(missing)
            + "；请先运行 python scripts/pull_sw2021_industry.py"
        )
    mapping = pd.read_parquet(mapping_path)
    catalog = pd.read_parquet(catalog_path)
    expected = int(config["d6"]["expected_industry_count"])
    if len(catalog) != expected:
        raise ValueError(f"行业目录为 {len(catalog)} 个，配置期望 {expected} 个")
    return mapping, catalog


def usable_style_dates(raw: pd.DataFrame, design_start, design_end) -> pd.DatetimeIndex:
    required = [
        "raw_Size",
        "raw_Beta",
        "raw_Momentum",
        "raw_ResidualVolatility",
        "raw_Liquidity",
    ]
    candidate = raw.loc[raw["trade_date"].between(design_start, design_end)]
    dates = []
    for date, group in candidate.groupby("trade_date", sort=True):
        if all(
            pd.to_numeric(group[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .nunique()
            >= 2
            for column in required
        ):
            dates.append(pd.Timestamp(date).normalize())
    return pd.DatetimeIndex(dates)


def assemble_exposures(
    weights: pd.DataFrame,
    raw_styles: pd.DataFrame,
    mapping: pd.DataFrame,
    catalog: pd.DataFrame,
    eligible_dates: pd.DatetimeIndex,
    *,
    winsor_mad_multiple: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    base = weights.loc[weights["trade_date"].isin(eligible_dates)].copy()
    base = base.rename(columns={"weight": "benchmark_weight"})
    base = base.merge(
        raw_styles,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        mapping[["instrument", "l1_code", "industry_name", "industry_factor"]],
        on="instrument",
        how="left",
        validate="many_to_one",
    )
    if base["industry_factor"].isna().any():
        missing = sorted(base.loc[base["industry_factor"].isna(), "instrument"].unique())
        raise FactorExposureError("行业映射连接后仍缺失：" + ", ".join(missing[:20]))

    exposure_frames = []
    audit_frames = []
    for date, group in base.groupby("trade_date", sort=True):
        processed, audit = process_style_cross_section(
            group,
            winsor_mad_multiple=float(winsor_mad_multiple),
        )
        audit.insert(0, "trade_date", pd.Timestamp(date).normalize())
        exposure_frames.append(processed)
        audit_frames.append(audit)
    if not exposure_frames:
        raise ValueError("没有生成任何 D6 风格暴露截面")
    exposures = pd.concat(exposure_frames, ignore_index=True)
    exposures, industry_columns = build_industry_dummies(exposures, catalog)
    style_columns = [f"style_{factor}" for factor in STYLE_FACTORS]
    return (
        exposures.sort_values(["trade_date", "instrument"]).reset_index(drop=True),
        pd.concat(audit_frames, ignore_index=True),
        industry_columns,
        style_columns,
    )


def audit_weekly_matrices(
    exposures: pd.DataFrame,
    weekly_dates: pd.DatetimeIndex,
    industry_columns: list[str],
    style_columns: list[str],
    *,
    normalization_tolerance: float,
    industry_exposure_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    benchmark_rows = []
    factor_columns = industry_columns + style_columns
    selected = exposures.loc[exposures["trade_date"].isin(weekly_dates)]
    for date, group in selected.groupby("trade_date", sort=True):
        audit = audit_factor_matrix(
            group,
            industry_columns,
            style_columns,
            normalization_tolerance=float(normalization_tolerance),
            industry_exposure_tolerance=float(industry_exposure_tolerance),
        )
        audit_rows.append({"trade_date": pd.Timestamp(date).normalize(), **audit})
        factor_exposure = benchmark_factor_exposure(group, factor_columns)
        benchmark_rows.extend(
            {
                "trade_date": pd.Timestamp(date).normalize(),
                "factor": factor,
                "factor_type": "industry" if factor in industry_columns else "style",
                "benchmark_exposure": float(value),
            }
            for factor, value in factor_exposure.items()
        )
    if not audit_rows:
        raise ValueError("没有可审计的周频 D6 暴露矩阵")
    return pd.DataFrame(audit_rows), pd.DataFrame(benchmark_rows)


def save_covariance(estimate, reports_dir: Path) -> tuple[Path, Path, dict]:
    matrix_path = reports_dir / "d6_covariance_latest.npz"
    temporary = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            covariance=estimate.covariance,
            instruments=np.asarray(estimate.instruments),
            as_of_date=str(estimate.as_of_date.date()),
        )
    temporary.replace(matrix_path)
    eigenvalues = np.linalg.eigvalsh(estimate.covariance)
    metadata = {
        "as_of_date": str(estimate.as_of_date.date()),
        "estimator": "sklearn.covariance.LedoitWolf",
        "frequency": "daily",
        "annualization_factor_for_te": 252,
        "observations": estimate.observations,
        "included_instruments": len(estimate.instruments),
        "excluded_instruments": list(estimate.excluded_instruments),
        "excluded_count": len(estimate.excluded_instruments),
        "shrinkage": estimate.shrinkage,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "usage": "ex-ante tracking-error reporting only; not an optimization constraint",
        "matrix_file": str(matrix_path.relative_to(ROOT)),
    }
    metadata_path = reports_dir / "d6_covariance_metadata.json"
    atomic_json(metadata, metadata_path)
    return matrix_path, metadata_path, metadata


def write_report(summary: dict, reports_dir: Path) -> None:
    acceptance = summary["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标"
    audit = summary["matrix_audit"]
    covariance = summary["covariance"]
    lines = [
        "# D6 因子暴露矩阵报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 固定口径",
        "",
        "- 行业：固定 SW2021 申万一级分类，31 列 one-hot；按整体计划明确采用非 PIT 版本，不做历史回填。",
        "- 风格：Size、Non-linear Size、Beta、Momentum、Residual Volatility、Liquidity，共 6 列。",
        "- 截面处理：3×MAD 去极值 → 总市值加权去均值 → 缺失值中性填补为 0 → 等权总体标准差归一。",
        "- Non-linear Size：Size³ 在总市值权重下对 Size 与截距正交化。",
        "- Beta 与残差波动率使用过去 252 个交易日相对 H00905.CSI 全收益指数的滚动回归矩；Momentum 为 T−252 至 T−21 的收益；Liquidity 为 21 个交易日平均换手率的自然对数。",
        "- Ledoit–Wolf Σ 只供后续报告事前 TE，不进入中性化约束、不调 λ、不做收益归因。",
        "",
        "## 暴露矩阵",
        "",
        f"- 输出区间：{summary['exposures']['start']} 至 {summary['exposures']['end']}。",
        f"- 日截面：{summary['exposures']['dates']:,} 个；行数：{summary['exposures']['rows']:,}。",
        f"- 列数：{summary['factors']['total']}（行业 {summary['factors']['industry']} + 风格 {summary['factors']['style']}）。",
        f"- 周频验收截面：{audit['weekly_dates']:,} 个，按当日有效列全部满秩：{'是' if acceptance['full_rank'] else '否'}。",
        f"- 最小有效矩阵秩：{audit['minimum_rank']} / {audit['minimum_effective_columns']}；对应最少 {audit['minimum_active_industries']} 个当日有效行业。",
        f"- 全局固定结构仍为 {summary['factors']['total']} 列；缺席行业保留为全零列，但不作为当日可估计维度参与秩验收。",
        f"- 行业哑变量行和最大误差：{audit['industry_row_sum_max_error']:.3e}。",
        f"- 风格总市值加权均值最大绝对值：{audit['style_market_cap_weighted_mean_max_abs']:.3e}。",
        f"- 风格等权标准差相对 1 的最大误差：{audit['style_equal_weight_std_max_error']:.3e}。",
        f"- 指数行业暴露与直接行业权重最大误差：{audit['benchmark_industry_exposure_max_error']:.3e}。",
        "",
        "说明：验收中的“均值≈0”按计划的“市值加权去均值”解释为总市值加权均值；普通等权均值不强制为 0。",
        "",
        "## 数据覆盖与已知局限",
        "",
        f"- 行业映射覆盖设计期成分股：{summary['industry']['mapped_instruments']:,}/{summary['industry']['required_instruments']:,}。",
        f"- 设计窗口覆盖从 {summary['exposures']['start']} 开始，与首个基准权重日一致：{'是' if acceptance['design_window_coverage'] else '否'}。",
        "- 252 日预热数据只用于生成 2014 年起点的滚动特征，不纳入设计期评价；预热数据仍严格早于 2019-01-01。",
        f"- 风格原始值最低截面覆盖率：{summary['style_quality']['minimum_raw_coverage']:.2%}；缺失值仅做中性暴露填补，并在明细中逐日记录。",
        "- 固定 SW2021 分类会把 2021 年前股票按同一版分类回看；它可用于一致的行业约束，但历史行业归因只能视为近似。",
        "",
        "## Ledoit–Wolf 协方差",
        "",
        f"- 截止日：{covariance['as_of_date']}；过去 {covariance['observations']} 个收益观察。",
        f"- 纳入 {covariance['included_instruments']} 只，因历史不足排除 {covariance['excluded_count']} 只。",
        f"- 收缩强度：{covariance['shrinkage']:.6f}；最小特征值：{covariance['minimum_eigenvalue']:.3e}。",
        "- 存储为日频协方差；后续事前年化 TE 使用 252 倍年化。",
        "",
        "## 输出",
        "",
        "- `reports/d6_factor_exposures.parquet`",
        "- `reports/d6_factor_audit.parquet`",
        "- `reports/d6_style_imputation.parquet`",
        "- `reports/d6_benchmark_exposures.parquet`",
        "- `reports/d6_covariance_latest.npz` 与 `reports/d6_covariance_metadata.json`",
        "- `reports/d6_summary.json`",
        "",
        "## 下一步",
        "",
        "D6 通过后进入 D7：用本矩阵对行业与风格偏离施加线性硬约束；Σ 仅计算并报告事前 TE。",
        "",
    ]
    path = reports_dir / "D6因子暴露矩阵报告.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = load_config()
    d6 = config["d6"]
    reports_dir = ROOT / config["data"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D6 必须在 design 模式运行，不得解封 OOS 数据")
    d5_summary = require_passed_d5(reports_dir)
    if config["industry"].get("pit") is not False or d6["industry_version"] != "SW2021":
        raise ValueError("D6 必须使用计划写死的固定非 PIT SW2021 行业分类")
    configured_styles = tuple(d6["style_factors"])
    if configured_styles != STYLE_FACTORS:
        raise ValueError(f"D6 六个风格因子必须固定为 {list(STYLE_FACTORS)}")

    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    history_start = design_start - pd.Timedelta(days=int(d6["history_buffer_calendar_days"]))
    processed_dir = Path(config["data"]["processed_design_dir"])
    weights = access.read_parquet(
        processed_dir / "benchmark_weights.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    weights["trade_date"] = parse_date_series(weights["trade_date"])
    instruments = sorted(weights["instrument"].astype(str).unique())
    mapping, catalog = require_industry_files(config)
    require_industry_coverage(instruments, mapping)

    print("读取 D6 所需行情、H00905.CSI 与 daily_basic……", flush=True)
    close = load_close_prices(access, instruments, history_start, design_end)
    benchmark = load_benchmark_returns(access, config, history_start, design_end)
    daily_basic = load_daily_basic(
        access, config, instruments, history_start, design_end
    )
    raw_styles = compute_raw_style_history(
        close,
        benchmark,
        daily_basic,
        beta_window=int(d6["beta_window_trade_days"]),
        beta_minimum_observations=int(d6["beta_minimum_observations"]),
        momentum_window=int(d6["momentum_window_trade_days"]),
        momentum_skip=int(d6["momentum_skip_trade_days"]),
        liquidity_window=int(d6["liquidity_window_trade_days"]),
        liquidity_minimum_observations=int(d6["liquidity_minimum_observations"]),
    )
    eligible_dates = usable_style_dates(raw_styles, design_start, design_end)
    eligible_dates = eligible_dates.intersection(pd.DatetimeIndex(weights["trade_date"].unique()))
    if eligible_dates.empty:
        raise ValueError(
            "滚动因子没有形成可用设计期截面；请检查 H00905.CSI 与 Qlib 是否含足够的设计期前历史"
        )
    print(
        f"可用暴露区间：{eligible_dates.min().date()} 至 {eligible_dates.max().date()}，"
        f"{len(eligible_dates)} 个交易日",
        flush=True,
    )

    exposures, style_audit, industry_columns, style_columns = assemble_exposures(
        weights,
        raw_styles,
        mapping,
        catalog,
        eligible_dates,
        winsor_mad_multiple=float(d6["winsor_mad_multiple"]),
    )
    weekly_dates = select_weekly_sample_dates(weights)
    weekly_dates = weekly_dates.intersection(eligible_dates)
    matrix_audit, benchmark_exposures = audit_weekly_matrices(
        exposures,
        weekly_dates,
        industry_columns,
        style_columns,
        normalization_tolerance=float(d6["normalization_tolerance"]),
        industry_exposure_tolerance=float(d6["industry_exposure_tolerance"]),
    )

    last_date = pd.Timestamp(exposures["trade_date"].max())
    latest_instruments = exposures.loc[
        exposures["trade_date"].eq(last_date), "instrument"
    ].tolist()
    returns = (
        close.pivot(index="trade_date", columns="instrument", values="close")
        .pct_change(fill_method=None)
        .stack(future_stack=True)
        .rename("return")
        .reset_index()
    )
    covariance = estimate_ledoit_wolf_covariance(
        returns,
        as_of_date=last_date,
        instruments=latest_instruments,
        window=int(d6["covariance_window_trade_days"]),
        minimum_observations=int(d6["covariance_minimum_observations"]),
    )
    _, _, covariance_metadata = save_covariance(covariance, reports_dir)

    first_weight_date = pd.Timestamp(weights["trade_date"].min()).normalize()
    first_exposure_date = pd.Timestamp(exposures["trade_date"].min()).normalize()

    acceptance = {
        "upstream_d5": bool(d5_summary["acceptance"]["passed"]),
        "design_window_coverage": bool(first_exposure_date == first_weight_date),
        "full_rank": bool(matrix_audit["full_rank"].all()),
        "style_normalization": bool(
            matrix_audit["style_market_cap_weighted_mean_max_abs"].max()
            <= float(d6["normalization_tolerance"])
            and matrix_audit["style_equal_weight_std_max_error"].max()
            <= float(d6["normalization_tolerance"])
        ),
        "industry_one_hot": bool(
            matrix_audit["industry_row_sum_max_error"].max()
            <= float(d6["industry_exposure_tolerance"])
        ),
        "benchmark_industry_exposure": bool(
            matrix_audit["benchmark_industry_exposure_max_error"].max()
            <= float(d6["industry_exposure_tolerance"])
        ),
        "covariance_positive_semidefinite": bool(
            covariance_metadata["minimum_eigenvalue"] >= -1e-12
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    summary = {
        "mode": access.mode,
        "upstream_d5_passed": True,
        "oos_data_accessed": False,
        "industry": {
            "classification": "申万一级 SW2021",
            "pit": False,
            "expected_industries": int(d6["expected_industry_count"]),
            "actual_industries": len(industry_columns),
            "required_instruments": len(instruments),
            "mapped_instruments": int(mapping["instrument"].isin(instruments).sum()),
        },
        "factors": {
            "industry": len(industry_columns),
            "style": len(style_columns),
            "total": len(industry_columns) + len(style_columns),
            "industry_columns": industry_columns,
            "style_columns": style_columns,
        },
        "exposures": {
            "start": str(exposures["trade_date"].min().date()),
            "end": str(exposures["trade_date"].max().date()),
            "dates": int(exposures["trade_date"].nunique()),
            "rows": int(len(exposures)),
            "history_buffer_start": str(history_start.date()),
        },
        "style_quality": {
            "minimum_raw_coverage": float(style_audit["raw_coverage"].min()),
            "maximum_neutral_imputation_rate": float(
                1.0 - style_audit["raw_coverage"].min()
            ),
        },
        "matrix_audit": {
            "weekly_dates": int(len(matrix_audit)),
            "minimum_rank": int(matrix_audit["matrix_rank"].min()),
            "minimum_effective_columns": int(matrix_audit["columns"].min()),
            "minimum_active_industries": int(
                matrix_audit["active_industry_columns"].min()
            ),
            "minimum_global_rank": int(matrix_audit["global_matrix_rank"].min()),
            "maximum_inactive_industries": int(
                len(industry_columns)
                - matrix_audit["active_industry_columns"].min()
            ),
            "industry_row_sum_max_error": float(
                matrix_audit["industry_row_sum_max_error"].max()
            ),
            "style_market_cap_weighted_mean_max_abs": float(
                matrix_audit["style_market_cap_weighted_mean_max_abs"].max()
            ),
            "style_equal_weight_std_max_error": float(
                matrix_audit["style_equal_weight_std_max_error"].max()
            ),
            "benchmark_industry_exposure_max_error": float(
                matrix_audit["benchmark_industry_exposure_max_error"].max()
            ),
        },
        "covariance": covariance_metadata,
        "acceptance": acceptance,
    }

    keep = [
        "trade_date",
        "instrument",
        "benchmark_weight",
        "market_cap",
        "l1_code",
        "industry_name",
        "industry_factor",
        *industry_columns,
        *style_columns,
    ]
    atomic_parquet(exposures[keep], reports_dir / "d6_factor_exposures.parquet")
    atomic_parquet(matrix_audit, reports_dir / "d6_factor_audit.parquet")
    atomic_parquet(style_audit, reports_dir / "d6_style_imputation.parquet")
    atomic_parquet(
        benchmark_exposures, reports_dir / "d6_benchmark_exposures.parquet"
    )
    atomic_json(summary, reports_dir / "d6_summary.json")
    write_report(summary, reports_dir)

    print(f"D6 验收：{'达标' if acceptance['passed'] else '未达标'}")
    print(
        f"暴露矩阵：{len(industry_columns)} 个行业 + {len(style_columns)} 个风格；"
        f"周频最小有效秩 {int(matrix_audit['matrix_rank'].min())}/"
        f"{int(matrix_audit['columns'].min())}"
    )
    print(f"协方差：{len(covariance.instruments)} 只股票，收缩强度 {covariance.shrinkage:.6f}")
    print(f"报告：{reports_dir / 'D6因子暴露矩阵报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

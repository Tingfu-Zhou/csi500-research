from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d3_core import assert_feature_timestamps, make_label_windows
from src.d4_core import (
    build_forward_excess_labels,
    compute_ic_decay,
    cross_sectional_zscore,
    decile_return_table,
    evaluate_acceptance,
    icir_by_period,
    rank_ic_series,
    select_weekly_sample_dates,
    split_design_samples,
    summarize_ic,
    validate_alpha158_spec,
)
from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_calendar(
    access: ResearchDataAccess, config: dict, start_date, end_date
) -> pd.DatetimeIndex:
    relative = Path(config["data"]["tushare_raw_dir"]) / "trade_cal.parquet"
    frame = access.read_parquet(
        relative,
        date_col="cal_date",
        start_date=start_date,
        end_date=end_date,
    )
    frame["cal_date"] = parse_date_series(frame["cal_date"])
    dates = frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"]
    calendar = pd.DatetimeIndex(dates).normalize().sort_values().unique()
    if calendar.empty:
        raise ValueError("设计窗口交易日历为空")
    return calendar


def load_alpha158_features(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date,
    end_date,
    expected_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.contrib.data.handler import Alpha158DL

    # Qlib 0.9.7 exposes Alpha158.get_feature_config as an instance method;
    # Alpha158DL is its canonical loader and provides the same 158-field config
    # without constructing a data handler (which would load data eagerly).
    fields, names = Alpha158DL.get_feature_config()
    validate_alpha158_spec(fields, names, expected_count)
    raw = access.features(instruments, list(fields), start_date, end_date)
    if raw.empty:
        raise ValueError("Qlib 没有返回 Alpha158 特征")
    if raw.shape[1] != len(names):
        raise ValueError(f"Qlib 特征列数 {raw.shape[1]} 与配置 {len(names)} 不一致")
    raw = raw.copy()
    raw.columns = list(names)
    features = raw.reset_index().rename(columns={"datetime": "sample_date"})
    if not {"instrument", "sample_date"}.issubset(features.columns):
        raise ValueError(f"Qlib 特征索引字段异常：{list(features.columns[:3])}")
    features["sample_date"] = pd.to_datetime(features["sample_date"]).dt.normalize()
    features[list(names)] = features[list(names)].replace([np.inf, -np.inf], np.nan)
    if features.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("Alpha158 特征存在重复的 日期×股票")
    spec = pd.DataFrame(
        {"feature_number": np.arange(1, len(names) + 1), "feature": names, "expression": fields}
    )
    return (
        features.sort_values(["sample_date", "instrument"]).reset_index(drop=True),
        spec,
    )


def load_vwap(
    access: ResearchDataAccess, instruments: list[str], start_date, end_date
) -> pd.DataFrame:
    raw = access.features(instruments, ["$vwap"], start_date, end_date)
    if raw.empty:
        raise ValueError("Qlib 没有返回 VWAP")
    raw = raw.copy()
    raw.columns = ["vwap"]
    frame = raw.reset_index().rename(columns={"datetime": "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["vwap"] = pd.to_numeric(frame["vwap"], errors="coerce")
    return frame.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def fit_default_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    seed: int,
    model_path: Path,
):
    import lightgbm as lgb

    train = train.copy()
    validation = validation.copy()
    train["model_target"] = cross_sectional_zscore(train, "excess_return")
    usable_train = train["model_target"].notna()
    if not usable_train.any():
        raise ValueError("训练目标在截面排序后全部缺失")
    all_missing = [name for name in feature_names if train.loc[usable_train, name].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"训练期存在全缺失 Alpha158 特征：{all_missing[:10]}")

    # 只固定随机种子；树数、叶子数、学习率等预测超参保持安装版本默认值。
    model = lgb.LGBMRegressor(random_state=int(seed))
    model.fit(train.loc[usable_train, feature_names], train.loc[usable_train, "model_target"])
    validation["prediction"] = model.predict(validation[feature_names])
    if not np.isfinite(validation["prediction"]).all():
        raise ValueError("LightGBM 产生非有限预测")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    model.booster_.save_model(str(temporary))
    temporary.replace(model_path)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    params = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in model.get_params().items()
    }
    return model, validation, importance.reset_index(drop=True), params, lgb.__version__


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    temporary.replace(path)


def plot_diagnostics(
    rank_ic: pd.DataFrame,
    period_icir: pd.DataFrame,
    decay: pd.DataFrame,
    deciles: pd.DataFrame,
    figures_dir: Path,
    icir_threshold: float,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    paths = {
        "rank_ic_timeseries": figures_dir / "d4_rank_ic_timeseries.png",
        "icir": figures_dir / "d4_icir.png",
        "ic_decay": figures_dir / "d4_ic_decay.png",
        "decile_returns": figures_dir / "d4_decile_returns.png",
    }

    series = rank_ic.copy()
    series["rolling_13w"] = series["rank_ic"].rolling(13, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(series["sample_date"], series["rank_ic"], color="#8aa1b4", lw=0.8, label="Weekly rank IC")
    ax.plot(series["sample_date"], series["rolling_13w"], color="#d24a43", lw=2.0, label="13-week mean")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set(title="Validation rank IC", xlabel="Signal date", ylabel="Spearman IC")
    ax.legend(frameon=False)
    save_figure(fig, paths["rank_ic_timeseries"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(period_icir["period"], period_icir["icir"], color=["#7f9db9", "#7f9db9", "#d24a43"])
    ax.axhline(icir_threshold, color="#333333", ls="--", lw=1.0, label=f"Threshold {icir_threshold:.2f}")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="Rank ICIR by period", xlabel="Period", ylabel="ICIR (mean / std)")
    ax.legend(frameon=False)
    save_figure(fig, paths["icir"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(decay["horizon"], decay["rank_ic_mean"], marker="o", color="#d24a43", lw=2)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(decay["horizon"])
    ax.set(title="Rank IC decay", xlabel="Forward VWAP return horizon (trading days)", ylabel="Mean rank IC")
    save_figure(fig, paths["ic_decay"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(deciles["decile"], deciles["mean_excess_return"], color="#7f9db9")
    bars[-1].set_color("#d24a43")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(deciles["decile"])
    ax.set(title="Validation decile monotonicity", xlabel="Prediction decile (low to high)", ylabel="Mean 5-day excess return")
    save_figure(fig, paths["decile_returns"])
    plt.close(fig)
    return {key: str(path.relative_to(ROOT)) for key, path in paths.items()}


def write_report(summary: dict, deciles: pd.DataFrame, reports_dir: Path) -> None:
    acceptance = summary["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标（结果有效，按计划不调参）"
    metrics = summary["validation_metrics"]
    monotonicity = summary["decile_monotonicity"]
    thresholds = summary["thresholds"]
    lines = [
        "# D4 Alpha158 + LightGBM 打分基线报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 固定口径",
        "",
        f"- 特征：Qlib {summary['versions']['qlib']} 的 Alpha158，共 {summary['features']['used']} 列。",
        f"- 模型：LightGBM {summary['versions']['lightgbm']} `LGBMRegressor`；除随机种子外使用安装版本默认超参，未早停、未搜索、未调参。",
        "- 训练目标：原始 5 日超额收益按 Alpha158 默认处理方式，在每个周截面内做 z-score；所有验收指标仍对原始超额收益计算。",
        "- 标签：信号日 T 后第一个交易日 VWAP 买入，T+6 交易日 VWAP 卖出；基准为信号日 `w_b` 对相同个股收益加权。",
        "- 采样：每个 W-FRI 周期的最后一个实际交易日，因此节假日周可能不是周五。",
        f"- 标签覆盖率：只做审计，不设计划外硬门槛；观测中位数 {summary['label_quality']['median_observed_coverage']:.2%}，最低 {summary['label_quality']['minimum_observed_coverage']:.2%}。",
        "",
        "## 时间切分与 purge",
        "",
        f"- 训练：{summary['samples']['train_start']} 至 {summary['samples']['train_end']}，{summary['samples']['train_rows']:,} 行 / {summary['samples']['train_dates']:,} 个截面。",
        f"- 验证：{summary['samples']['validation_start']} 至 {summary['samples']['validation_end']}，{summary['samples']['validation_rows']:,} 行 / {summary['samples']['validation_dates']:,} 个截面。",
        f"- 训练期末因标签跨界而剔除：{summary['samples']['purged_train_dates']:,} 个周截面。",
        f"- 训练集最后标签结束日：{summary['samples']['last_train_label_end']}，不晚于训练截止日。",
        "- 验证数据未参与训练或早停选择。",
        "",
        "## 验证结果",
        "",
        f"- Rank IC 均值：{metrics['rank_ic_mean']:.4f}（门槛 > {thresholds['rank_ic_mean_min']:.2f}）——{'通过' if acceptance['rank_ic_mean'] else '未通过'}",
        f"- ICIR：{metrics['icir']:.4f}（定义为周截面 IC 均值 / 标准差；门槛 > {thresholds['icir_min']:.2f}）——{'通过' if acceptance['icir'] else '未通过'}",
        f"- 十分组相邻上升：{monotonicity['adjacent_up_steps']}/{monotonicity['possible_adjacent_steps']}（门槛 ≥ {thresholds['monotonic_min_adjacent_steps']}，且 D10>D1）——{'通过' if acceptance['decile_monotonicity'] else '未通过'}",
        f"- D10−D1 五日超额：{monotonicity['top_minus_bottom']:.2%}。",
        f"- 十分组均值与组号 Spearman：{monotonicity['decile_spearman']:.4f}。",
        "",
        "## 十分组收益",
        "",
        "| 组别 | 平均未来5日超额 | 样本数 |",
        "|---:|---:|---:|",
    ]
    for row in deciles.itertuples(index=False):
        lines.append(f"| D{int(row.decile)} | {row.mean_excess_return:.3%} | {int(row.observations):,} |")
    lines.extend(
        [
            "",
            "## 图表与审计输出",
            "",
            "- `reports/figures/d4_rank_ic_timeseries.png`",
            "- `reports/figures/d4_icir.png`",
            "- `reports/figures/d4_ic_decay.png`",
            "- `reports/figures/d4_decile_returns.png`",
            "- `reports/d4_validation_predictions.parquet`",
            "- `reports/d4_rank_ic.parquet`、`d4_ic_decay.parquet`、`d4_decile_returns.parquet`",
            "- `reports/d4_alpha158_spec.parquet`、`d4_feature_importance.parquet`、`d4_label_coverage.parquet`",
            "- `models/d4_lightgbm_baseline.txt`、`reports/d4_summary.json`",
            "",
            "> D4 只判断打分基线是否过门槛。LightGBM 分数仍是无量纲值，收益量纲校准属于 D5，不能在本日提前完成。",
        ]
    )
    path = reports_dir / "D4打分基线报告.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    d4 = config["d4"]
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D4 必须在 design 模式运行，不得解封 OOS 数据")

    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    train_start = pd.Timestamp(d4["train_start"])
    train_end = pd.Timestamp(d4["train_end"])
    validation_start = pd.Timestamp(d4["validation_start"])
    validation_end = pd.Timestamp(d4["validation_end"])
    configured_horizon = int(config["label"]["horizon_trade_days"])
    decay_horizons = sorted({int(item) for item in d4["ic_decay_horizons"]})
    horizons = sorted(set(decay_horizons) | {configured_horizon})
    reports_dir = ROOT / config["data"]["reports_dir"]
    figures_dir = ROOT / config["data"]["figures_dir"]
    model_path = ROOT / config["data"]["models_dir"] / "d4_lightgbm_baseline.txt"

    calendar = load_calendar(access, config, design_start, design_end)
    processed_dir = Path(config["data"]["processed_design_dir"])
    weights = access.read_parquet(
        processed_dir / "benchmark_weights.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    weights["trade_date"] = pd.to_datetime(weights["trade_date"]).dt.normalize()
    sample_dates = select_weekly_sample_dates(weights)
    instruments = sorted(weights["instrument"].unique())

    print("读取 Qlib Alpha158 特征……", flush=True)
    features, feature_spec = load_alpha158_features(
        access,
        instruments,
        design_start,
        design_end,
        int(d4["expected_feature_count"]),
    )
    feature_names = feature_spec["feature"].tolist()
    features = features.loc[features["sample_date"].isin(sample_dates)].copy()
    print(f"Alpha158 周截面：{features['sample_date'].nunique()} 个", flush=True)

    vwap = load_vwap(access, instruments, design_start, design_end)
    labels, label_coverage = build_forward_excess_labels(
        vwap,
        weights,
        sample_dates,
        calendar,
        horizons,
    )
    main_labels = labels.loc[labels["horizon"].eq(configured_horizon)].copy()
    main_coverage = label_coverage.loc[
        label_coverage["horizon"].eq(configured_horizon)
    ].copy()
    coverage_by_year = (
        main_coverage.assign(year=main_coverage["sample_date"].dt.year)
        .groupby("year")
        .agg(
            weekly_windows=("sample_date", "size"),
            median_weight_coverage=("weight_coverage", "median"),
            minimum_weight_coverage=("weight_coverage", "min"),
            median_valid_labels=("valid_label_count", "median"),
        )
    )
    print("5 日标签权重覆盖率（年度）：", flush=True)
    print(coverage_by_year.to_string(), flush=True)
    dataset = main_labels.merge(
        features,
        on=["sample_date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError("Alpha158 特征与 5 日标签没有交集")
    dataset[feature_names] = dataset[feature_names].replace([np.inf, -np.inf], np.nan)
    dataset_year_counts = (
        dataset.groupby(dataset["sample_date"].dt.year).size().astype(int).to_dict()
    )
    print(f"可用 5 日标签样本年度分布：{dataset_year_counts}", flush=True)

    # 实际 Alpha158 特征的最大信息时点就是信号日 T；复用 D3 时间戳断言。
    main_windows = make_label_windows(sample_dates, calendar, configured_horizon)
    valid_metadata = dataset[["sample_date", "instrument"]].copy()
    valid_metadata["feature_max_timestamp"] = valid_metadata["sample_date"]
    assert_feature_timestamps(valid_metadata, main_windows)

    attempted_train_dates = pd.DatetimeIndex(
        dataset.loc[dataset["sample_date"].between(train_start, train_end), "sample_date"].unique()
    )
    train, validation = split_design_samples(
        dataset,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    purged_train_dates = len(
        attempted_train_dates.difference(pd.DatetimeIndex(train["sample_date"].unique()))
    )

    print("训练默认 LightGBM……", flush=True)
    _, validation_predictions, importance, model_params, lightgbm_version = fit_default_lightgbm(
        train,
        validation,
        feature_names,
        int(config["project"]["random_seed"]),
        model_path,
    )
    # Spearman 相关系数的数学下限是两只股票；不另设计划外样本门槛。
    min_cross_section = 2
    rank_ic = rank_ic_series(
        validation_predictions, min_cross_section_size=min_cross_section
    )
    ic_summary = summarize_ic(rank_ic["rank_ic"])
    period_icir = icir_by_period(rank_ic)
    decay_labels = labels.loc[
        labels["sample_date"].between(validation_start, validation_end)
    ].copy()
    decay = compute_ic_decay(
        validation_predictions[["sample_date", "instrument", "prediction"]],
        decay_labels,
        decay_horizons,
        min_cross_section_size=min_cross_section,
    )
    decile_table, monotonicity = decile_return_table(
        validation_predictions,
        deciles=int(d4["deciles"]),
    )
    thresholds = d4["acceptance"]
    acceptance = evaluate_acceptance(
        ic_summary,
        monotonicity,
        rank_ic_mean_min=float(thresholds["rank_ic_mean_min"]),
        icir_min=float(thresholds["icir_min"]),
        monotonic_min_adjacent_steps=int(thresholds["monotonic_min_adjacent_steps"]),
    )
    figure_paths = plot_diagnostics(
        rank_ic,
        period_icir,
        decay,
        decile_table,
        figures_dir,
        float(thresholds["icir_min"]),
    )

    import qlib

    coverage_used = label_coverage.loc[label_coverage["horizon"].isin(horizons)]
    summary = {
        "mode": access.mode,
        "versions": {"qlib": qlib.__version__, "lightgbm": lightgbm_version},
        "feature_set": d4["feature_set"],
        "features": {"requested": len(feature_spec), "used": len(feature_names)},
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "parameter_policy": "installed-version defaults; random_state fixed only",
            "parameters": model_params,
            "target_transform": "weekly cross-sectional z-score (Alpha158 default CSZScoreNorm semantics)",
            "model_file": str(model_path.relative_to(ROOT)),
        },
        "label": {
            "type": "excess_return",
            "entry": "T+1_vwap",
            "exit": f"T+{configured_horizon + 1}_vwap",
            "horizon_trade_days": configured_horizon,
            "benchmark": "signal-date w_b weighted constituent return over identical VWAP window",
        },
        "label_quality": {
            "minimum_observed_coverage": float(coverage_used["weight_coverage"].min()),
            "median_observed_coverage": float(coverage_used["weight_coverage"].median()),
            "hard_coverage_filter": False,
        },
        "samples": {
            "train_start": str(train["sample_date"].min().date()),
            "train_end": str(train["sample_date"].max().date()),
            "train_rows": len(train),
            "train_dates": int(train["sample_date"].nunique()),
            "validation_start": str(validation_predictions["sample_date"].min().date()),
            "validation_end": str(validation_predictions["sample_date"].max().date()),
            "validation_rows": len(validation_predictions),
            "validation_dates": int(validation_predictions["sample_date"].nunique()),
            "purged_train_dates": int(purged_train_dates),
            "last_train_label_end": str(train["label_end_timestamp"].max().date()),
        },
        "validation_metrics": ic_summary,
        "decile_monotonicity": monotonicity,
        "thresholds": {
            "rank_ic_mean_min": float(thresholds["rank_ic_mean_min"]),
            "icir_min": float(thresholds["icir_min"]),
            "monotonic_min_adjacent_steps": int(thresholds["monotonic_min_adjacent_steps"]),
        },
        "acceptance": acceptance,
        "figures": figure_paths,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(
        validation_predictions[
            [
                "sample_date",
                "instrument",
                "prediction",
                "excess_return",
                "stock_return",
                "benchmark_return",
                "label_start_timestamp",
                "label_end_timestamp",
                "weight_coverage",
            ]
        ],
        reports_dir / "d4_validation_predictions.parquet",
    )
    atomic_parquet(rank_ic, reports_dir / "d4_rank_ic.parquet")
    atomic_parquet(period_icir, reports_dir / "d4_icir_by_period.parquet")
    atomic_parquet(decay, reports_dir / "d4_ic_decay.parquet")
    atomic_parquet(decile_table, reports_dir / "d4_decile_returns.parquet")
    atomic_parquet(feature_spec, reports_dir / "d4_alpha158_spec.parquet")
    atomic_parquet(importance, reports_dir / "d4_feature_importance.parquet")
    atomic_parquet(label_coverage, reports_dir / "d4_label_coverage.parquet")
    atomic_json(summary, reports_dir / "d4_summary.json")
    write_report(summary, decile_table, reports_dir)

    print(f"D4 验收：{'达标' if acceptance['passed'] else '未达标（不调参）'}")
    print(f"报告：{reports_dir / 'D4打分基线报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

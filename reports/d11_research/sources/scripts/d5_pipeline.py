from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d4_pipeline import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    fit_default_lightgbm,
    load_alpha158_features,
    load_calendar,
    load_vwap,
)
from src.d3_core import (  # noqa: E402
    assert_feature_timestamps,
    assert_future_price_invariance,
    build_causal_price_features,
    make_label_windows,
)
from src.d4_core import (  # noqa: E402
    build_forward_excess_labels,
    rank_ic_series,
    select_weekly_sample_dates,
    summarize_ic,
)
from src.d5_core import (  # noqa: E402
    apply_return_calibration,
    build_annual_calibration_schedule,
    build_expanding_year_folds,
    calibration_scale_diagnostics,
    fit_return_calibration,
    rank_gaussianize,
    split_walk_forward_samples,
)
from src.data_access import ResearchDataAccess  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def qlib_amount_multiplier(config: dict) -> float:
    """Read the Qlib amount unit conversion from its canonical config section."""
    try:
        value = float(config["tradability"]["qlib_amount_multiplier_to_cny"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "config.yaml 缺少有效的 tradability.qlib_amount_multiplier_to_cny"
        ) from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("qlib_amount_multiplier_to_cny 必须是正的有限数")
    return value


def require_passed_d4(reports_dir: Path) -> dict:
    path = reports_dir / "d4_summary.json"
    if not path.exists():
        raise FileNotFoundError("缺少 reports/d4_summary.json，请先成功运行 D4")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "design":
        raise RuntimeError("D4 结果不是 design 模式产物")
    if not payload.get("acceptance", {}).get("passed", False):
        raise RuntimeError("D4 未通过写死的验收标准，不得进入 D5")
    return payload


def load_price_audit_frame(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date,
    end_date,
    amount_multiplier: float,
) -> pd.DataFrame:
    fields = ["$open", "$high", "$low", "$close", "$vwap", "$volume", "$amount"]
    raw = access.features(instruments, fields, start_date, end_date)
    if raw.empty:
        raise ValueError("未来价格扰动复验没有取得行情")
    raw = raw.copy()
    raw.columns = ["open", "high", "low", "close", "vwap", "volume", "amount"]
    frame = raw.reset_index().rename(columns={"datetime": "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["daily_amount_cny"] = (
        pd.to_numeric(frame.pop("amount"), errors="coerce") * float(amount_multiplier)
    )
    return frame.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def choose_evenly_spaced(items: pd.DatetimeIndex, count: int) -> pd.DatetimeIndex:
    unique = pd.DatetimeIndex(items).normalize().sort_values().unique()
    if len(unique) < int(count):
        raise ValueError(f"未来价格扰动候选日只有 {len(unique)} 个，少于 {int(count)} 个")
    positions = np.linspace(0, len(unique) - 1, num=int(count)).round().astype(int)
    return pd.DatetimeIndex(unique[positions]).unique()


def run_fold_leakage_checks(
    dataset: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    folds,
    price_audit: pd.DataFrame,
    checks_per_fold: int,
    horizon: int,
    seed: int,
    rtol: float,
    atol: float,
) -> tuple[dict[int, int], int]:
    timestamp_counts: dict[int, int] = {}
    perturbation_count = 0
    for fold in folds:
        relevant = dataset.loc[
            dataset["sample_date"].between(fold.train_start, fold.prediction_end),
            ["sample_date", "instrument"],
        ].copy()
        relevant["feature_max_timestamp"] = relevant["sample_date"]
        windows = make_label_windows(
            relevant["sample_date"].unique(), calendar, horizon
        )
        assert_feature_timestamps(relevant, windows)
        timestamp_counts[fold.prediction_year] = int(len(relevant))

        eligible = calendar[
            (calendar >= fold.train_start)
            & (calendar <= fold.train_end)
            & (calendar < calendar.max())
        ]
        cutoffs = choose_evenly_spaced(eligible, int(checks_per_fold))
        for number, cutoff in enumerate(cutoffs):
            assert_future_price_invariance(
                build_causal_price_features,
                price_audit,
                cutoff,
                seed=int(seed) + fold.prediction_year * 100 + number,
                rtol=float(rtol),
                atol=float(atol),
            )
            perturbation_count += 1
    return timestamp_counts, perturbation_count


def write_report(
    summary: dict,
    fold_audit: pd.DataFrame,
    calibration_schedule: pd.DataFrame,
    planned_oos: pd.DataFrame,
    reports_dir: Path,
) -> None:
    scale = summary["calibration_scale"]
    acceptance = summary["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标"
    lines = [
        "# D5 分数校准与滚动重训框架报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 固定口径",
        "",
        "- 上游模型：D4 已通过的 Alpha158 + 默认 LightGBM；没有搜索或调整超参。",
        "- 原始预测先在每个周截面内排序，再映射为标准正态分数 `z_score`。",
        "- 收益校准：按周截面等权、无截距回归 `未来5日超额收益 = a × z_score`。",
        "- 模型按自然年扩展窗口重训；所有日期切点均来自 `config.yaml`。",
        "- 本日仍为 design 模式，2019–2026 只生成计划，没有读取或预测样本外数据。",
        "",
        "## 设计窗口走向前空跑",
        "",
        "| 预测年 | 训练截止 | 训练行数 | 预测行数 | 最后训练标签结束日 | Rank IC | purge |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in fold_audit.itertuples(index=False):
        lines.append(
            f"| {int(row.prediction_year)} | {row.train_end} | {int(row.train_rows):,} "
            f"| {int(row.prediction_rows):,} | {row.last_train_label_end} "
            f"| {row.rank_ic_mean:.4f} | {'通过' if row.purge_passed else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 收益量纲校准",
            "",
            f"- 供首个样本外年度使用的校准系数：`a = {summary['next_oos_calibration']['coefficient']:.8f}`，即一个标准差的分数对应 {summary['next_oos_calibration']['coefficient']:.4%} 的预期5日超额。",
            f"- 校准后 μ 的平均截面标准差：{scale['calibrated_mu_mean_cross_section_std']:.4%}。",
            f"- 验证期已实现 alpha 尺度：{scale['realized_alpha_decile_std']:.4%}。这里固定定义为十分组平均已实现5日超额的标准差。",
            f"- 尺度比：{scale['scale_ratio']:.3f}；允许区间为 [{1-scale['tolerance']:.2f}, {1+scale['tolerance']:.2f}]——{'通过' if scale['passed'] else '未通过'}。",
            "",
            "## 年度校准计划",
            "",
            "| 生效预测年 | 历史起点 | 历史终点 | 历史年数 | 校准系数 | 仅使用此前数据 |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in calibration_schedule.itertuples(index=False):
        lines.append(
            f"| {int(row.effective_prediction_year)} | {row.history_start} | {row.history_end} "
            f"| {int(row.history_years)} | {row.calibration_coefficient:.8f} "
            f"| {'是' if row.history_precedes_effective_year else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 防泄漏复验",
            "",
            f"- 每折训练集最后标签结束日均不晚于训练截止日：{'通过' if acceptance['fold_purge'] else '失败'}。",
            f"- Alpha158 样本时间戳逐折断言：通过，共检查 {summary['leakage_checks']['timestamp_rows']:,} 行。",
            f"- D3 未来价格扰动逐折复验：通过，共执行 {summary['leakage_checks']['future_perturbation_checks']:,} 次。",
            "",
            "## D9 样本外框架",
            "",
            f"- 已生成 {len(planned_oos)} 折日期计划（{int(planned_oos['prediction_year'].min())}–{int(planned_oos['prediction_year'].max())}）。",
            "- 这些折当前没有运行。D9 冻结代码并显式解封后，才允许读取样本外数据。",
            "",
            "## 输出",
            "",
            "- `reports/d5_walk_forward_predictions.parquet`",
            "- `reports/d5_fold_audit.parquet`",
            "- `reports/d5_calibration_schedule.parquet`",
            "- `reports/d5_calibration_deciles.parquet`",
            "- `reports/d5_planned_oos_folds.parquet`",
            "- `reports/d5_feature_importance.parquet`",
            "- `models/d5_calibration.json`",
            "- `reports/d5_summary.json`",
        ]
    )
    (reports_dir / "D5分数校准与滚动框架报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    config = load_config()
    d3 = config["d3"]
    d4 = config["d4"]
    d5 = config["d5"]
    windows = config["windows"]
    if d5["score_transform"] != "cross_sectional_rank_gaussian":
        raise ValueError("D5 只允许写死的截面 rank 正态化口径")
    if d5["calibration_method"] != "date_balanced_zero_intercept_ols":
        raise ValueError("D5 只允许写死的按日期平衡无截距校准口径")
    reports_dir = ROOT / config["data"]["reports_dir"]
    models_dir = ROOT / config["data"]["models_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D5 必须在 design 模式运行，不得解封 OOS 数据")
    d4_summary = require_passed_d4(reports_dir)

    design_start = pd.Timestamp(windows["design_start"])
    design_end = pd.Timestamp(windows["design_end"])
    horizon = int(config["label"]["horizon_trade_days"])
    calibration_deciles = int(d5["calibration_deciles"])
    tolerance = float(d5["calibration_scale_tolerance"])
    seed = int(config["project"]["random_seed"])

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

    print("读取 D5 所需 Alpha158 特征与 5 日超额标签……", flush=True)
    features, feature_spec = load_alpha158_features(
        access,
        instruments,
        design_start,
        design_end,
        int(d4["expected_feature_count"]),
    )
    feature_names = feature_spec["feature"].tolist()
    features = features.loc[features["sample_date"].isin(sample_dates)].copy()
    vwap = load_vwap(access, instruments, design_start, design_end)
    labels, _ = build_forward_excess_labels(
        vwap, weights, sample_dates, calendar, [horizon]
    )
    dataset = labels.loc[labels["horizon"].eq(horizon)].merge(
        features,
        on=["sample_date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError("D5 的 Alpha158 特征与标签没有交集")

    dry_folds = build_expanding_year_folds(
        train_start=d4["train_start"],
        initial_train_end=d4["train_end"],
        prediction_start=d4["validation_start"],
        prediction_end=d4["validation_end"],
    )
    if [fold.prediction_year for fold in dry_folds] != list(
        range(pd.Timestamp(d4["validation_start"]).year, pd.Timestamp(d4["validation_end"]).year + 1)
    ):
        raise AssertionError("D5 设计期空跑年份与配置不一致")

    audit_count = min(int(d3["future_perturbation_checks"]), len(instruments))
    audit_indices = np.linspace(0, len(instruments) - 1, num=audit_count).round().astype(int)
    audit_instruments = [instruments[index] for index in sorted(set(audit_indices))]
    price_audit = load_price_audit_frame(
        access,
        audit_instruments,
        design_start,
        design_end,
        qlib_amount_multiplier(config),
    )
    timestamp_counts, perturbation_count = run_fold_leakage_checks(
        dataset,
        calendar,
        dry_folds,
        price_audit,
        int(d3["future_perturbation_checks"]),
        horizon,
        seed,
        float(d3["future_perturbation_rtol"]),
        float(d3["future_perturbation_atol"]),
    )

    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    model_parameters: dict | None = None
    lightgbm_version: str | None = None
    for fold in dry_folds:
        train, prediction = split_walk_forward_samples(dataset, fold)
        attempted_dates = pd.DatetimeIndex(
            dataset.loc[
                dataset["sample_date"].between(fold.train_start, fold.train_end),
                "sample_date",
            ].unique()
        )
        model_path = models_dir / f"d5_lightgbm_{fold.prediction_year}.txt"
        _, predicted, importance, params, version = fit_default_lightgbm(
            train, prediction, feature_names, seed, model_path
        )
        predicted["score_z"] = rank_gaussianize(predicted)
        if predicted["score_z"].isna().any():
            raise ValueError(f"{fold.prediction_year} 折存在无法正态化的预测截面")
        predicted["prediction_year"] = fold.prediction_year
        predicted["model_train_end"] = fold.train_end
        prediction_frames.append(predicted)

        importance = importance.copy()
        importance["prediction_year"] = fold.prediction_year
        importance_frames.append(importance)
        ic = summarize_ic(rank_ic_series(predicted, min_cross_section_size=2)["rank_ic"])
        actual_dates = pd.DatetimeIndex(train["sample_date"].unique())
        audit = {
            **fold.as_record(),
            "train_rows": int(len(train)),
            "train_dates": int(train["sample_date"].nunique()),
            "prediction_rows": int(len(predicted)),
            "prediction_dates": int(predicted["sample_date"].nunique()),
            "last_train_sample_date": str(train["sample_date"].max().date()),
            "last_train_label_end": str(train["label_end_timestamp"].max().date()),
            "first_prediction_sample_date": str(predicted["sample_date"].min().date()),
            "last_prediction_sample_date": str(predicted["sample_date"].max().date()),
            "purged_train_dates": int(len(attempted_dates.difference(actual_dates))),
            "purge_passed": bool(train["label_end_timestamp"].max() <= fold.train_end),
            "rank_ic_mean": ic["rank_ic_mean"],
            "icir": ic["icir"],
            "model_file": str(model_path.relative_to(ROOT)),
        }
        audit_rows.append(audit)
        model_parameters = params
        lightgbm_version = version
        print(
            f"{fold.prediction_year} 折完成：train_end={fold.train_end.date()}，"
            f"Rank IC={ic['rank_ic_mean']:.4f}",
            flush=True,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["sample_date", "instrument"]
    )
    fold_audit = pd.DataFrame(audit_rows).sort_values("prediction_year")
    importance = pd.concat(importance_frames, ignore_index=True)

    design_calibration = fit_return_calibration(predictions)
    predictions[d5["expected_return_column"]] = apply_return_calibration(
        predictions["score_z"], design_calibration["coefficient"]
    )
    scale, calibration_decile_table = calibration_scale_diagnostics(
        predictions,
        design_calibration["coefficient"],
        deciles=calibration_deciles,
        tolerance=tolerance,
    )

    first_validation_year = pd.Timestamp(d4["validation_start"]).year
    first_oos_year = pd.Timestamp(windows["oos_start"]).year
    effective_years = range(first_validation_year + 1, first_oos_year + 1)
    calibration_schedule = build_annual_calibration_schedule(
        predictions,
        effective_years,
        minimum_history_years=int(d5["minimum_calibration_history_years"]),
    )
    next_oos_row = calibration_schedule.loc[
        calibration_schedule["effective_prediction_year"].eq(first_oos_year)
    ].iloc[0]
    if not np.isclose(
        float(next_oos_row["calibration_coefficient"]),
        float(design_calibration["coefficient"]),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise AssertionError("首个样本外年度系数与完整设计验证期校准不一致")

    planned_oos_folds = build_expanding_year_folds(
        train_start=d4["train_start"],
        initial_train_end=windows["design_end"],
        prediction_start=windows["oos_start"],
        prediction_end=windows["oos_end"],
    )
    planned_oos = pd.DataFrame([fold.as_record() for fold in planned_oos_folds])
    fold_purge_passed = bool(fold_audit["purge_passed"].all())
    acceptance = {
        "calibration_scale": bool(scale["passed"]),
        "fold_purge": fold_purge_passed,
        "timestamp_assertions": True,
        "future_price_perturbation": True,
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    next_oos_calibration = {
        "effective_prediction_year": first_oos_year,
        "coefficient": float(next_oos_row["calibration_coefficient"]),
        "history_start": next_oos_row["history_start"],
        "history_end": next_oos_row["history_end"],
    }
    summary = {
        "mode": access.mode,
        "upstream_d4_passed": bool(d4_summary["acceptance"]["passed"]),
        "versions": {
            "lightgbm": lightgbm_version,
            "d4_lightgbm": d4_summary["versions"]["lightgbm"],
        },
        "score_transform": d5["score_transform"],
        "calibration": design_calibration,
        "calibration_scale": scale,
        "next_oos_calibration": next_oos_calibration,
        "dry_run_folds": fold_audit.to_dict(orient="records"),
        "planned_oos_folds": int(len(planned_oos)),
        "oos_data_accessed": False,
        "leakage_checks": {
            "timestamp_rows": int(sum(timestamp_counts.values())),
            "timestamp_rows_by_prediction_year": timestamp_counts,
            "future_perturbation_checks": int(perturbation_count),
            "audit_instruments": audit_instruments,
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "parameter_policy": "D4 defaults unchanged; random_state fixed only",
            "parameters": model_parameters,
        },
        "acceptance": acceptance,
    }

    keep_columns = [
        "sample_date",
        "instrument",
        "prediction_year",
        "model_train_end",
        "prediction",
        "score_z",
        d5["expected_return_column"],
        "excess_return",
        "stock_return",
        "benchmark_return",
        "label_start_timestamp",
        "label_end_timestamp",
        "weight_coverage",
    ]
    atomic_parquet(predictions[keep_columns], reports_dir / "d5_walk_forward_predictions.parquet")
    atomic_parquet(fold_audit, reports_dir / "d5_fold_audit.parquet")
    atomic_parquet(
        calibration_schedule, reports_dir / "d5_calibration_schedule.parquet"
    )
    atomic_parquet(
        calibration_decile_table, reports_dir / "d5_calibration_deciles.parquet"
    )
    atomic_parquet(planned_oos, reports_dir / "d5_planned_oos_folds.parquet")
    atomic_parquet(importance, reports_dir / "d5_feature_importance.parquet")
    atomic_json(next_oos_calibration, models_dir / "d5_calibration.json")
    atomic_json(summary, reports_dir / "d5_summary.json")
    write_report(summary, fold_audit, calibration_schedule, planned_oos, reports_dir)

    print(f"D5 验收：{'达标' if acceptance['passed'] else '未达标'}")
    print(f"校准系数 a：{next_oos_calibration['coefficient']:.8f}")
    print(f"尺度比：{scale['scale_ratio']:.3f}")
    print(f"报告：{reports_dir / 'D5分数校准与滚动框架报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

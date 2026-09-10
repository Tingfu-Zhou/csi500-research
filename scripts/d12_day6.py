"""Day 6: P2 evidence review, next-week R2 freeze and forward-start checklist.

Run after the completed local Day-5 output::

    python -u scripts/d12_day6.py --run-id day6_v1 --day5-run-id day5_v1

This command reconstructs Alpha158 only for descriptive feature-distribution review.
It never fits LightGBM, changes decile cut-points, simulates another portfolio, or runs R2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d11_pipeline import load_weekly_alpha158
from scripts.d12_day4 import (load_handoff, run_preflight, save_snapshot,
                              validate_frozen_spec)
from src.d12_day4_io import require_day4_run
from src.d12_day6 import (TARGET_YEARS, constraint_tradability_review,
                          evidence_matrix, feature_distribution_review,
                          feature_extreme_observations, forward_startup_checklist,
                          monthly_concentration,
                          r2_readiness, signal_year_diagnostics,
                          validate_day6_freeze)
from src.d12_day6_io import (Day6Output, assert_snapshot_unchanged,
                             require_day5_run, snapshot_inputs)
from src.d12_repair_access import RepairedDataAccess, repaired_config
from src.d12_repo import D12Error, sha256


UNRESOLVED_CLASS_A = ["P0-12 membership timing", "P0-24 alias missing prices",
                      "P0-27 unclassified factor breaks"]


def say(message):
    print(message, flush=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sources(day5_run_id, r1_run_id):
    day5_path, day5_manifest = require_day5_run(ROOT, day5_run_id)
    day5_summary = load_json(day5_path / "summary.json")
    if (day5_summary.get("historical_decision") != "No-Go"
            or day5_summary.get("decision_changed") is not False
            or day5_summary.get("model_fitted") is not False
            or day5_summary.get("r2_run") is not False):
        raise D12Error("Day-5 summary does not preserve the registered research boundary")
    parity = day5_summary.get("nominal_r1_replay", {})
    if parity.get("passed") is not True or any(
            float(value) != 0. for value in parity.get("maximum_errors", {}).values()):
        raise D12Error("Day-5 nominal R1 replay did not reproduce every saved state")
    r1_path, r1_manifest = require_day4_run(ROOT, r1_run_id)
    source = day5_summary.get("source_identity", {})
    if (source.get("r1_run_id") != r1_run_id
            or source.get("r1_manifest_sha256") != sha256(r1_path / "manifest.json")):
        raise D12Error("Day-5 and Day-4 R1 identities differ")
    return day5_path, day5_manifest, day5_summary, r1_path, r1_manifest


def reconstruct_feature_matrix(handoff, config, predictions):
    repaired = repaired_config(
        config, processed_design_dir=handoff["design"].relative_to(ROOT),
        processed_oos_dir=handoff["oos"].relative_to(ROOT))
    predictions = predictions.copy()
    predictions["sample_date"] = pd.to_datetime(predictions.sample_date)
    relevant = predictions.loc[predictions.sample_date.dt.year.between(2019, 2025)].copy()
    dates = pd.DatetimeIndex(sorted(relevant.sample_date.unique()))
    instruments = sorted(relevant.instrument.astype(str).unique())
    if not len(dates) or not instruments:
        raise D12Error("No repaired prediction keys available for the Day-6 feature review")
    access = RepairedDataAccess(ROOT, handoff["provider"])
    say("D12 Day6 reconstructing Alpha158 for 2019-2025; no model fitting")
    features, specification = load_weekly_alpha158(access, instruments, dates, repaired)
    saved_specification = pd.read_parquet(handoff["run"] / "feature_spec.parquet")
    try:
        pd.testing.assert_frame_equal(
            specification.reset_index(drop=True), saved_specification.reset_index(drop=True),
            check_dtype=False, check_like=True)
    except AssertionError as error:
        raise D12Error("Reconstructed Alpha158 feature specification changed") from error
    names = saved_specification.feature.astype(str).tolist()
    keys = ["sample_date", "instrument"]
    features["sample_date"] = pd.to_datetime(features.sample_date)
    matrix = relevant[keys + ["feature_available"]].merge(
        features[keys + names], on=keys, how="left", validate="one_to_one")
    reconstructed_available = matrix[names].apply(
        pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any(axis=1)
    if not reconstructed_available.equals(matrix.feature_available.astype(bool)):
        mismatches = int(reconstructed_available.ne(matrix.feature_available.astype(bool)).sum())
        raise D12Error(f"Reconstructed Alpha158 availability differs on {mismatches} rows")
    return matrix.drop(columns="feature_available"), saved_specification, {
        "provider": str(handoff["provider"].relative_to(ROOT)),
        "feature_spec_sha256": sha256(handoff["run"] / "feature_spec.parquet"),
        "features": len(names), "rows": int(len(matrix)),
        "dates": int(matrix.sample_date.nunique()),
        "years": sorted(matrix.sample_date.dt.year.unique().astype(int).tolist()),
        "model_fitted": False,
    }


def save_frame(output, name, frame, *, parquet=True):
    output.table(name, frame)
    if parquet:
        output.parquet(name + ".parquet", frame)


def write_report(output, day5_summary, signal, feature_summary, constraints,
                 readiness, readiness_summary, concentration):
    performance = day5_summary["performance"]
    r0 = performance["repaired_r0_nominal"]
    r1 = performance["r1_nominal"]
    r0x = performance["repaired_r0_impact2x"]
    r1x = performance["r1_impact2x"]
    signal = signal.set_index("year")
    flags = feature_summary.set_index("year").review_flags.to_dict()
    binding = constraints.groupby(["strategy", "year"]).turnover_cap_binding_fraction.first()
    blocked = readiness.loc[~readiness.status.eq("passed"), "condition"].tolist()
    best = concentration.set_index("strategy")
    text = f"""# D12 第六天：P2 初诊、R2 冻结与前向启动清单

状态：本报告已由本机结果生成；最终完成状态以同目录 `manifest.json` 为准。未训练模型、未重跑组合、未运行 R2。

## 第五天证据收口

| 路径 | 年化超额 | IR | NW-t 最小值 | 最大相对回撤 |
|---|---:|---:|---:|---:|
| 修复 R0 名义 | {r0['annualized_excess_return']:.4%} | {r0['information_ratio']:.4f} | {r0['newey_west_t_minimum']:.4f} | {r0['maximum_relative_drawdown']:.4%} |
| R1 名义 | {r1['annualized_excess_return']:.4%} | {r1['information_ratio']:.4f} | {r1['newey_west_t_minimum']:.4f} | {r1['maximum_relative_drawdown']:.4%} |
| 修复 R0 冲击 ×2 | {r0x['annualized_excess_return']:.4%} | {r0x['information_ratio']:.4f} | {r0x['newey_west_t_minimum']:.4f} | {r0x['maximum_relative_drawdown']:.4%} |
| R1 冲击 ×2 | {r1x['annualized_excess_return']:.4%} | {r1x['information_ratio']:.4f} | {r1x['newey_west_t_minimum']:.4f} | {r1x['maximum_relative_drawdown']:.4%} |

R1 名义超额比修复 R0 低 {(r1['annualized_excess_return']-r0['annualized_excess_return']):.4%}，
但冲击翻倍造成的超额损失较小：R0 {(r0x['annualized_excess_return']-r0['annualized_excess_return']):.4%}，
R1 {(r1x['annualized_excess_return']-r1['annualized_excess_return']):.4%}。两者均只通过 3/6 数值门，
归因残差质量仍未通过；历史 No-Go 不变。

## P2 初步诊断

- 2023：平均 Rank IC {signal.loc[2023, 'rank_ic_mean']:.5f}，十分组顶减底实现超额均值 {signal.loc[2023, 'top_minus_bottom_mean']:.4%}，相邻上升 {int(signal.loc[2023, 'adjacent_up_steps'])}/9。
- 2025：平均 Rank IC {signal.loc[2025, 'rank_ic_mean']:.5f}，十分组顶减底实现超额均值 {signal.loc[2025, 'top_minus_bottom_mean']:.4%}，相邻上升 {int(signal.loc[2025, 'adjacent_up_steps'])}/9。
- Alpha158 描述性漂移筛查标记：2023 年 {int(flags[2023])} 项、2025 年 {int(flags[2025])} 项。标记只要求人工复核，不能自动判成数据错误或训练稳健性问题。
- 主动换手上限触及率：R0 2023/2025 为 {binding.loc[('repaired_r0', 2023)]:.2%}/{binding.loc[('repaired_r0', 2025)]:.2%}；R1 为 {binding.loc[('r1', 2023)]:.2%}/{binding.loc[('r1', 2025)]:.2%}。这是观测埋点，不是放松约束的反事实收益。
- 最佳五个月占全部正月度超额：原 D11 {best.loc['original_d11', 'best_five_share_of_positive_excess']:.2%}、修复 R0 {best.loc['repaired_r0', 'best_five_share_of_positive_excess']:.2%}、R1 {best.loc['r1_nominal', 'best_five_share_of_positive_excess']:.2%}。

## R2 与前向结论

R2 状态：**{readiness_summary['decision']}**，本周没有运行。未通过/待核条件：{'; '.join(blocked)}。
折数口径已冻结为 2017–2026 共 10 个年度训练模型，其中 2017–2018 仅建立因果校准历史，
2019–2026 是 8 个评分/交易年度；下次启动条件复核日为 2026-09-16。

前向记录仍是计划，不是已部署任务：9/14 冻结最小规格、9/21 计划启动、12/21 首次约三个月复核。
默认 R0；R1 若记录只能走独立影子路径。没有成交引擎时只保存分数与模拟订单，不声称策略收益；
2026-08-31 之后若新增撮合，必须先建立独立的前向有效税费配置，历史配置保持冻结。

详细证据见 `p2_signal_year_summary.tsv`、`p2_signal_deciles.tsv`、
`p2_feature_distribution.tsv`、`p2_feature_extreme_observations.tsv`、
`p2_constraint_tradability.tsv`、`r2_start_conditions.tsv`、
`forward_startup_checklist.tsv` 与 `evidence_matrix.tsv`。
"""
    output.text("day6_report.md", text)


def run_stage(args, output):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    experiments = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text(
        encoding="utf-8"))
    repository = run_preflight(output, config)
    r1_checks = validate_frozen_spec(config, experiments)
    day6_checks = validate_day6_freeze(experiments)
    output.json("frozen_spec_identity.json", {
        "r1": r1_checks, "day6": day6_checks,
        "experiments_sha256": sha256(ROOT / "config/d12_experiments.yaml"),
        "registry_sha256": sha256(ROOT / "reports/d12/experiment_registry.tsv")})
    handoff = load_handoff()
    day5_path, day5_manifest, day5_summary, r1_path, r1_manifest = load_sources(
        args.day5_run_id, args.r1_run_id)
    source_identity = {
        "day5_run_id": args.day5_run_id,
        "day5_manifest_sha256": sha256(day5_path / "manifest.json"),
        "r1_run_id": args.r1_run_id,
        "r1_manifest_sha256": sha256(r1_path / "manifest.json"),
        "repaired_r0": str(handoff["run"].relative_to(ROOT)),
        "repaired_r0_manifest_sha256": sha256(handoff["run"] / "repaired_r0_manifest.json"),
        "corrected_provider": str(handoff["provider"].relative_to(ROOT)),
    }
    output.json("source_identity.json", source_identity)

    before = snapshot_inputs(ROOT, config)
    before_summary = save_snapshot(output, "inputs_before", before)
    predictions = pd.read_parquet(handoff["run"] / "predictions.parquet")
    labels = pd.read_parquet(handoff["run"] / "labels_for_training_and_evaluation.parquet")
    matrix, feature_spec, reconstruction = reconstruct_feature_matrix(
        handoff, config, predictions)
    output.json("feature_reconstruction.json", reconstruction)

    deciles, by_date, yearly, signal_summary = signal_year_diagnostics(
        predictions, labels, config["windows"]["oos_end"])
    feature_detail, feature_summary = feature_distribution_review(
        matrix, feature_spec.feature.astype(str).tolist())
    feature_extremes = feature_extreme_observations(matrix, feature_detail)
    save_frame(output, "p2_signal_deciles", deciles)
    save_frame(output, "p2_signal_by_date", by_date)
    save_frame(output, "p2_signal_year_summary", yearly)
    output.json("p2_signal_summary.json", signal_summary)
    save_frame(output, "p2_feature_distribution", feature_detail)
    save_frame(output, "p2_feature_year_summary", feature_summary)
    save_frame(output, "p2_feature_extreme_observations", feature_extremes)

    repaired = repaired_config(
        config, processed_design_dir=handoff["design"].relative_to(ROOT),
        processed_oos_dir=handoff["oos"].relative_to(ROOT))
    constraint_frames = []
    for strategy, path in [("repaired_r0", handoff["run"]), ("r1", r1_path)]:
        constraint_frames.append(constraint_tradability_review(
            pd.read_parquet(path / "targets.parquet"),
            pd.read_parquet(path / "audits.parquet"),
            pd.read_parquet(path / "trades.parquet"), repaired, strategy))
    constraints = pd.concat(constraint_frames, ignore_index=True)
    save_frame(output, "p2_constraint_tradability", constraints)

    monthly = pd.read_csv(r1_path / "monthly_three_way_comparison.tsv", sep="\t")
    concentration = monthly_concentration(monthly)
    save_frame(output, "best_month_concentration", concentration)
    evidence = evidence_matrix(day5_summary)
    save_frame(output, "evidence_matrix", evidence)
    readiness, readiness_summary = r2_readiness(
        UNRESOLVED_CLASS_A, day5_completed=day5_manifest.get("status") == "completed")
    forward = forward_startup_checklist(experiments)
    save_frame(output, "r2_start_conditions", readiness, parquet=False)
    output.json("r2_decision.json", readiness_summary)
    save_frame(output, "forward_startup_checklist", forward, parquet=False)
    write_report(output, day5_summary, yearly, feature_summary, constraints,
                 readiness, readiness_summary, concentration)

    summary = {
        "classification": "day6_p2_initial_diagnostic_r2_and_forward_freeze",
        "source_identity": source_identity,
        "model_fitted": False,
        "portfolio_replayed": False,
        "r2_run": False,
        "target_years": list(TARGET_YEARS),
        "p2": {"signal": yearly.to_dict("records"),
               "feature_review": feature_summary.to_dict("records"),
               "constraints": constraints.to_dict("records"),
               "automatic_flags_are_causal_findings": False,
               "manual_review_required": True},
        "r2": readiness_summary,
        "forward": {"deployed": False, "planned_start": "2026-09-21",
                    "first_review_target": "2026-12-21"},
        "historical_decision": "No-Go",
        "decision_changed": False,
        "new_oos_validation": False,
        "comparison_status": "provisional_due_to_unresolved_P0_12",
        "unresolved": UNRESOLVED_CLASS_A,
    }
    output.json("summary.json", summary)

    after = snapshot_inputs(ROOT, config)
    after_summary = save_snapshot(output, "inputs_after", after)
    assert_snapshot_unchanged(before, after)
    if before_summary["canonical_json_sha256"] != after_summary["canonical_json_sha256"]:
        raise D12Error("Day-6 input snapshot digest changed")
    return {
        "mode": "day6_p2_evidence_and_freeze", "inputs_unchanged": True,
        "historical_decision": "No-Go", "decision_changed": False,
        "new_oos_validation": False, "model_fitted": False,
        "portfolio_replayed": False, "r2_run": False,
        "r2_decision": readiness_summary["decision"],
        "comparison_status": summary["comparison_status"],
        "source_identity": source_identity, "repository": repository,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--day5-run-id", default="day5_v1")
    parser.add_argument("--r1-run-id", default="r1_nominal_v1")
    args = parser.parse_args()
    output = Day6Output(ROOT, args.run_id)
    started = time.perf_counter()
    manifest = {"run_id": args.run_id, "status": "running", "inputs_unchanged": False}
    try:
        result = run_stage(args, output)
        manifest.update(result, status="completed")
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        output.text("failure.txt", traceback.format_exc())
        say(f"{type(error).__name__}: {error}")
    manifest.update(recorded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    elapsed_seconds=time.perf_counter() - started,
                    output_sha256=output.hashes())
    output.json("manifest.json", manifest)
    say(f"Wrote {output.path}; status={manifest['status']}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

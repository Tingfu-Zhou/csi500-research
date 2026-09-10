"""Day 5: repaired R0/R1 diagnostics and execution-side impact x2 stress.

Run only after ``zero_v1`` and ``r1_nominal_v1`` completed locally::

    python -u scripts/d12_day5.py --run-id day5_v1 \
        --zero-run-id zero_v1 --r1-run-id r1_nominal_v1

This script never fits a model. It reuses repaired-R0 predictions, first replays the
saved nominal R1 path, then doubles execution impact with actual cash/holdings feedback.
R1's decision-side coefficient stays fixed at 0.5. R2 is outside this command.
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

from scripts.d11_diagnose import load_inputs as load_diagnostic_inputs, read_saved
from scripts.d12_day4 import (load_handoff, run_preflight, save_snapshot,
                              validate_frozen_spec)
from src.d3_core import CostModel
from src.d10_core import linked_monthly, require_close
from src.d11_core import simulate_causal
from src.d11_diagnostics import (COST_COLUMNS, attribute_d11, execution_diagnostics,
                                 performance_and_costs, reconstruct_d11,
                                 signal_diagnostics)
from src.d12_day4_io import require_day4_run
from src.d12_day5 import (attribution_comparison, daily_from_simulation,
                          impact2x_configuration, nominal_replay_parity,
                          performance_matrix, reference_gates, signal_calibration,
                          stress_effects)
from src.d12_day5_io import Day5Output, assert_snapshot_unchanged, snapshot_inputs
from src.d12_repair_access import RepairedDataAccess, repaired_config
from src.d12_repo import D12Error, sha256
from src.d12_r1 import simulate_r1


R0_FRAMES = ("daily", "trades", "orders", "targets", "audits", "predictions",
             "labels_for_training_and_evaluation", "factor_exposures")
PATH_FRAMES = ("daily", "trades", "orders", "targets", "audits")


def say(message):
    print(message, flush=True)


def load_sources(handoff, zero_run_id, r1_run_id):
    zero_path, zero_manifest = require_day4_run(ROOT, zero_run_id)
    r1_path, r1_manifest = require_day4_run(ROOT, r1_run_id)
    if (zero_manifest.get("mode") != "zero_gate" or zero_manifest.get("decision_k") != 0.
            or not zero_manifest.get("zero_gate", {}).get("passed")):
        raise D12Error("Day 5 requires a completed zero-coefficient gate")
    if (r1_manifest.get("mode") != "r1_nominal" or r1_manifest.get("decision_k") != .5
            or r1_manifest.get("zero_gate_run_id") != zero_run_id):
        raise D12Error("Day 5 requires the registered nominal R1 linked to the supplied gate")
    if Path(r1_manifest.get("repaired_r0", "")) != handoff["run"].relative_to(ROOT):
        raise D12Error("Day-4 R1 and Day-5 repaired R0 identities differ")
    for name in R0_FRAMES:
        if not (handoff["run"] / f"{name}.parquet").is_file():
            raise D12Error(f"Repaired R0 is missing {name}.parquet")
    r0 = {name: read_saved(handoff["run"] / f"{name}.parquet") for name in R0_FRAMES}
    r1 = {name: read_saved(r1_path / f"{name}.parquet") for name in PATH_FRAMES}
    return r0, r1, {
        "zero_run_id": zero_run_id,
        "zero_manifest_sha256": sha256(zero_path / "manifest.json"),
        "r1_run_id": r1_run_id,
        "r1_manifest_sha256": sha256(r1_path / "manifest.json"),
        "r1_performance": r1_manifest["performance"],
    }


def save_simulation(output, prefix, result, daily):
    output.parquet(prefix + "_daily.parquet", daily)
    for name in ["targets", "orders", "trades", "audits"]:
        output.parquet(prefix + f"_{name}.parquet", getattr(result, name))


def pnl_components(daily, path):
    ledger = daily[["trade_date", "portfolio_return", "benchmark_return"]].merge(
        path.daily, on="trade_date", validate="one_to_one")
    ledger["benchmark_component"] = -ledger.benchmark_return
    for name in COST_COLUMNS:
        ledger[name] = -ledger[name] / ledger.previous_nav
    columns = ["holding_return", "execution_return", "benchmark_component", *COST_COLUMNS]
    monthly = linked_monthly(ledger, columns)
    require_close(monthly[columns].sum(axis=1), monthly.excess_return,
                  "Day-5 monthly PnL components")
    return monthly


def diagnose_path(name, saved, market, daily_exposures, industries, styles,
                  config, output, *, attribution=True):
    capital = float(config["d8"]["main_capital_cny"])
    performance, monthly, yearly = performance_and_costs(
        saved["daily"], capital, config["d9"]["newey_west_lags"])
    path = reconstruct_d11(saved["daily"], saved["trades"], market, capital,
                           CostModel.from_config(config))
    execution, by_decision = execution_diagnostics(
        saved["trades"], saved["targets"], saved["audits"], path)
    output.parquet(name + "_accounting.parquet", path.daily)
    output.parquet(name + "_monthly_performance.parquet", monthly)
    output.parquet(name + "_yearly_performance.parquet", yearly)
    output.parquet(name + "_execution_by_decision.parquet", by_decision)
    output.table(name + "_monthly_performance", monthly)
    output.table(name + "_yearly_performance", yearly)
    output.table(name + "_execution_by_decision", by_decision)
    components = pnl_components(saved["daily"], path)
    output.parquet(name + "_pnl_components_monthly.parquet", components)
    output.table(name + "_pnl_components_monthly", components)
    result = {"performance_detail": performance, "performance": performance["net"],
              "execution": execution, "accounting": {
                  "maximum_nav_error_cny": float(path.daily.nav_error_cny.max()),
                  "maximum_daily_residual": float(path.daily.accounting_residual.abs().max())}}
    if attribution:
        say(f"D12 Day5 attributing {name} with repaired lagged daily exposures")
        daily_attr, monthly_attr, factors, fits, summary = attribute_d11(
            saved["daily"], path, market, daily_exposures, industries, styles)
        for suffix, frame in [("attribution_daily", daily_attr),
                              ("attribution_monthly", monthly_attr),
                              ("factor_returns", factors),
                              ("regression_audits", fits)]:
            output.parquet(name + "_" + suffix + ".parquet", frame)
        output.table(name + "_attribution_monthly", monthly_attr)
        require_close([summary["annualized_contributions"]["excess_return"]],
                      [performance["net"]["annualized_excess_return"]],
                      name + " attribution total")
        result["attribution"] = summary
    return result


def simulation_progress(label, total):
    count = 0
    def progress(target, orders, audit):
        nonlocal count
        count += 1
        if count == 1 or count % 25 == 0 or count == total:
            say(f"D12 Day5 {label}: {count}/{total} decisions")
    return progress


def require_healthy_solver(label, result, expected_decisions):
    counts = result.audits.solver_status.value_counts().to_dict()
    fallback = int(result.audits.fallback_used.sum())
    if len(result.audits) != expected_decisions or counts != {"optimal": expected_decisions} or fallback:
        raise D12Error(f"{label} solver audit failed: decisions={len(result.audits)}, "
                       f"status={counts}, fallback={fallback}")
    return {"decisions": expected_decisions, "solver_status_counts": counts,
            "fallback_used": fallback}


def cost_table(scenarios):
    rows = []
    for scenario, item in scenarios.items():
        execution = item.get("execution")
        if not execution:
            continue
        rows.append({"scenario": scenario, **execution["costs_cny"],
                     "total_cost_cny": execution["total_cost_cny"],
                     "order_records": execution["order_records"],
                     "filled_records": execution["filled_records"],
                     "fill_fraction_at_signal_prices": execution["fill_fraction_at_signal_prices"],
                     "forced_exit_cost_cny": execution["forced_exit_cost_cny"]})
    return pd.DataFrame(rows)


def write_report(output, matrix, effects, attribution, gates, calibration):
    perf = matrix.set_index("scenario")
    r0, r1 = perf.loc["repaired_r0_nominal"], perf.loc["r1_nominal"]
    r0x, r1x = perf.loc["repaired_r0_impact2x"], perf.loc["r1_impact2x"]
    attr = attribution.pivot(index="measure", columns="strategy", values="value")
    passed = gates.groupby("strategy").passed.agg(["sum", "count"])
    text = f"""# D12 第五天：修复 R0 / R1 归因、交易损耗与冲击压力

状态：本机全量运行结果已写入本报告；最终完成状态以同目录 `manifest.json` 为准。固定修复 R0 预测；未训练模型；R2 未运行。

| 路径 | 决策 k | 执行 k | 年化超额 | IR | NW-t 最小值 | 最大相对回撤 |
|---|---:|---:|---:|---:|---:|---:|
| 修复 R0 名义 | 0.0 | 0.5 | {r0.annualized_excess_return:.4%} | {r0.information_ratio:.4f} | {r0.newey_west_t_minimum:.4f} | {r0.maximum_relative_drawdown:.4%} |
| R1 名义 | 0.5 | 0.5 | {r1.annualized_excess_return:.4%} | {r1.information_ratio:.4f} | {r1.newey_west_t_minimum:.4f} | {r1.maximum_relative_drawdown:.4%} |
| 修复 R0 压力 | 0.0 | 1.0 | {r0x.annualized_excess_return:.4%} | {r0x.information_ratio:.4f} | {r0x.newey_west_t_minimum:.4f} | {r0x.maximum_relative_drawdown:.4%} |
| R1 压力 | 0.5 | 1.0 | {r1x.annualized_excess_return:.4%} | {r1x.information_ratio:.4f} | {r1x.newey_west_t_minimum:.4f} | {r1x.maximum_relative_drawdown:.4%} |

## 归因与信号

- 修复 R0 费后特异性占比：{float(attr.loc['specific_share_net', 'repaired_r0']):.2%}；归因质量门：{bool(attr.loc['three_block_residual_passed', 'repaired_r0'])}。
- R1 费后特异性占比：{float(attr.loc['specific_share_net', 'r1']):.2%}；归因质量门：{bool(attr.loc['three_block_residual_passed', 'r1'])}。
- R0/R1 复用同一预测，Rank IC、十分组与校准结果按设计完全相同；不是 R1 的新增信号收益。
- 十分组最高减最低已实现 5 日超额：{calibration['top_minus_bottom_realized']:.4%}；相邻上升 {calibration['adjacent_up_steps']}/9。

## 验收解释

- 原六门槛事后数值对照：修复 R0 {int(passed.loc['repaired_r0', 'sum'])}/{int(passed.loc['repaired_r0', 'count'])}，R1 {int(passed.loc['r1', 'sum'])}/{int(passed.loc['r1', 'count'])}；归因质量另列，不与数值门合并。
- `stress_effects.tsv` 分开列 R1 策略差异与各自冲击翻倍损耗；压力路径不是事后从净值减费用。
- P0-12、P0-24、P0-27 仍未关闭，绝对业绩和桥接解释保持临时口径。
- 历史结论仍为 No-Go；2019–2026 已反复查看，本次不是新的样本外验证。

详细证据见 `performance_matrix.tsv`、`cost_comparison.tsv`、`attribution_comparison.tsv`、
`yearly_ic.tsv`、`signal_calibration_deciles.tsv`、`reference_gates.tsv` 及对应 Parquet。
"""
    output.text("day5_report.md", text)


def run_stage(args, output):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    experiments = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text(encoding="utf-8"))
    repository = run_preflight(output, config)
    frozen = validate_frozen_spec(config, experiments)
    output.json("frozen_spec_identity.json", frozen)
    handoff = load_handoff()
    repaired = repaired_config(
        config, processed_design_dir=handoff["design"].relative_to(ROOT),
        processed_oos_dir=handoff["oos"].relative_to(ROOT))
    r0, r1, source_identity = load_sources(
        handoff, args.zero_run_id, args.r1_run_id)
    output.json("source_identity.json", source_identity)

    before = snapshot_inputs(ROOT, config)
    before_summary = save_snapshot(output, "inputs_before", before)
    access = RepairedDataAccess(ROOT, handoff["provider"])
    market, daily_exposures, flags, calendar, official, industries, styles = (
        load_diagnostic_inputs(access, repaired, r0["factor_exposures"], output))
    constrained_styles = ["style_" + name for name in repaired["d7"]["constrained_style_factors"]]
    if not set(constrained_styles).issubset(styles):
        raise D12Error("Rebuilt daily exposures do not contain the constrained R1 styles")
    official_return = official.set_index("trade_date").benchmark_return
    for label, daily in [("repaired R0", r0["daily"]), ("R1", r1["daily"])]:
        require_close(daily.benchmark_return,
                      official_return.reindex(daily.trade_date), label + " official benchmark")

    expected_decisions = len(r0["audits"])
    nominal_estimator = impact2x_configuration(repaired)[1]
    say("D12 Day5 replaying nominal R1 before any pressure result")
    nominal_replay = simulate_r1(
        market, r0["predictions"], r0["factor_exposures"], flags, calendar, repaired,
        industries, constrained_styles, nominal_estimator,
        progress=simulation_progress("r1_nominal_replay", expected_decisions))
    nominal_daily = daily_from_simulation(
        nominal_replay, r0["daily"], repaired["d8"]["main_capital_cny"])
    parity = nominal_replay_parity(nominal_replay, nominal_daily, r1)
    parity.update(require_healthy_solver("R1 nominal replay", nominal_replay, expected_decisions))
    output.json("r1_nominal_replay_parity.json", parity)
    save_simulation(output, "r1_nominal_replay", nominal_replay, nominal_daily)

    say("D12 Day5 reconstructing and attributing repaired R0 nominal")
    r0_nominal = diagnose_path("repaired_r0_nominal", r0, market, daily_exposures,
                               industries, styles, repaired, output)
    say("D12 Day5 reconstructing and attributing R1 nominal")
    r1_nominal = diagnose_path("r1_nominal", r1, market, daily_exposures,
                               industries, styles, repaired, output)
    for metric in ["annualized_excess_return", "information_ratio",
                   "newey_west_t_minimum", "maximum_relative_drawdown"]:
        require_close([r0_nominal["performance"][metric]],
                      [handoff["r0_manifest"]["research"]["performance"][metric]],
                      "repaired R0 saved performance " + metric)
        require_close([r1_nominal["performance"][metric]],
                      [source_identity["r1_performance"][metric]],
                      "R1 saved performance " + metric)

    signal_ic, yearly_ic = signal_diagnostics(
        r0["predictions"], r0["labels_for_training_and_evaluation"],
        repaired["windows"]["oos_end"])
    calibration_table, calibration_summary = signal_calibration(
        r0["predictions"], r0["labels_for_training_and_evaluation"],
        repaired["windows"]["oos_end"], deciles=10)
    output.parquet("signal_ic.parquet", signal_ic)
    output.parquet("yearly_ic.parquet", yearly_ic)
    output.table("signal_ic", signal_ic)
    output.table("yearly_ic", yearly_ic)
    output.table("signal_calibration_deciles", calibration_table)
    output.json("signal_calibration_summary.json", calibration_summary)

    stressed, decision_estimator = impact2x_configuration(repaired)
    say("D12 Day5 running repaired R0 with execution impact k=1.0")
    r0_stress_result = simulate_causal(
        market, r0["predictions"], r0["factor_exposures"], flags, calendar, stressed,
        industries, constrained_styles,
        progress=simulation_progress("r0_impact2x", expected_decisions))
    require_healthy_solver("repaired R0 impact2x", r0_stress_result, expected_decisions)
    r0_stress_daily = daily_from_simulation(
        r0_stress_result, r0["daily"], stressed["d8"]["main_capital_cny"])
    save_simulation(output, "repaired_r0_impact2x", r0_stress_result, r0_stress_daily)
    r0_stress = diagnose_path(
        "repaired_r0_impact2x",
        {"daily": r0_stress_daily, "trades": r0_stress_result.trades,
         "orders": r0_stress_result.orders, "targets": r0_stress_result.targets,
         "audits": r0_stress_result.audits}, market, daily_exposures, industries,
        styles, stressed, output, attribution=False)

    say("D12 Day5 running R1 with decision k=0.5 and execution impact k=1.0")
    r1_stress_result = simulate_r1(
        market, r0["predictions"], r0["factor_exposures"], flags, calendar, stressed,
        industries, constrained_styles, decision_estimator,
        progress=simulation_progress("r1_impact2x", expected_decisions))
    require_healthy_solver("R1 impact2x", r1_stress_result, expected_decisions)
    if not r1_stress_result.audits.decision_impact_k.eq(.5).all():
        raise D12Error("R1 pressure changed the registered decision-side coefficient")
    r1_stress_daily = daily_from_simulation(
        r1_stress_result, r0["daily"], stressed["d8"]["main_capital_cny"])
    save_simulation(output, "r1_impact2x", r1_stress_result, r1_stress_daily)
    r1_stress = diagnose_path(
        "r1_impact2x",
        {"daily": r1_stress_daily, "trades": r1_stress_result.trades,
         "orders": r1_stress_result.orders, "targets": r1_stress_result.targets,
         "audits": r1_stress_result.audits}, market, daily_exposures, industries,
        styles, stressed, output, attribution=False)

    original = json.loads((ROOT / "reports/d12/repair/repair_v3/compare/comparison.json").read_text(
        encoding="utf-8"))["original_performance"]
    scenarios = {
        "original_d11_reference": {"decision_k": 0., "execution_k": .5,
                                   "performance": original},
        "repaired_r0_nominal": {"decision_k": 0., "execution_k": .5, **r0_nominal},
        "r1_nominal": {"decision_k": .5, "execution_k": .5, **r1_nominal},
        "repaired_r0_impact2x": {"decision_k": 0., "execution_k": 1., **r0_stress},
        "r1_impact2x": {"decision_k": .5, "execution_k": 1., **r1_stress},
    }
    matrix = performance_matrix(scenarios)
    effects = stress_effects(matrix)
    attributes = {"repaired_r0": r0_nominal["attribution"],
                  "r1": r1_nominal["attribution"]}
    attribution_table = attribution_comparison(attributes)
    gates = reference_gates(
        {"repaired_r0": r0_nominal["performance"], "r1": r1_nominal["performance"]},
        {"repaired_r0": r0_stress["performance"], "r1": r1_stress["performance"]},
        attributes, repaired)
    costs = cost_table(scenarios)
    output.table("performance_matrix", matrix)
    output.table("stress_effects", effects)
    output.table("cost_comparison", costs)
    output.table("attribution_comparison", attribution_table)
    output.table("reference_gates", gates)
    write_report(output, matrix, effects, attribution_table, gates, calibration_summary)

    summary = {
        "classification": "day5_repaired_r0_r1_diagnostics_and_dynamic_execution_impact2x",
        "source_identity": source_identity,
        "model_fitted": False,
        "r2_run": False,
        "nominal_r1_replay": parity,
        "stress_specification": {
            "r0": {"decision_k": 0., "execution_nominal_k": .5, "execution_stress_k": 1.},
            "r1": {"decision_k": .5, "execution_nominal_k": .5, "execution_stress_k": 1.},
            "actual_cash_holdings_feedback": True,
            "post_hoc_fee_subtraction": False},
        "performance": {name: item["performance"] for name, item in scenarios.items()},
        "execution": {name: item["execution"] for name, item in scenarios.items()
                      if item.get("execution")},
        "attribution": attributes,
        "signal": {"yearly_ic": yearly_ic.to_dict("records"),
                   "calibration": calibration_summary,
                   "r0_r1_predictions_identical": True},
        "reference_gates": gates.to_dict("records"),
        "historical_decision": "No-Go",
        "decision_changed": False,
        "new_oos_validation": False,
        "comparison_status": "provisional_due_to_unresolved_P0_12",
        "unresolved": ["P0-12 membership timing", "P0-24 alias missing prices",
                       "P0-27 excluded factor breaks"],
    }
    output.json("summary.json", summary)

    after = snapshot_inputs(ROOT, config)
    after_summary = save_snapshot(output, "inputs_after", after)
    assert_snapshot_unchanged(before, after)
    if before_summary["canonical_json_sha256"] != after_summary["canonical_json_sha256"]:
        raise D12Error("Day-5 input snapshot digest changed")
    return {"mode": "day5_diagnostics_and_stress", "inputs_unchanged": True,
            "historical_decision": "No-Go", "decision_changed": False,
            "new_oos_validation": False, "source_identity": source_identity,
            "performance": summary["performance"],
            "comparison_status": summary["comparison_status"],
            "repository": repository}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zero-run-id", default="zero_v1")
    parser.add_argument("--r1-run-id", default="r1_nominal_v1")
    args = parser.parse_args()
    output = Day5Output(ROOT, args.run_id)
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

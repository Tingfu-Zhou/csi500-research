"""Day 4: full zero-coefficient gate, then the one frozen nominal R1 candidate.

Examples (run in separate fresh processes so Qlib cannot retain another provider):

    python -u scripts/d12_day4.py --stage gate --run-id zero_v1
    python -u scripts/d12_day4.py --stage r1 --run-id r1_nominal_v1 --gate-run-id zero_v1

Both stages use the repaired ``repair_v3`` provider, rebuilt caches and repaired R0.
The R1 stage reuses the repaired R0 predictions; it does not fit LightGBM.  Attribution
and execution-side impact x2 belong to Day 5 and are deliberately absent here.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scripts import d12_preflight as preflight
from scripts.d11_diagnose import check_parity
from scripts.d11_pipeline import load_open_calendar, prepare_oos_market, read_frame
from scripts.d12_day3 import load_weights_flags
from src.d9_core import summarize_monthly_performance
from src.d12_core import ImpactEstimator
from src.d12_day4_io import (Day4Output, assert_snapshot_unchanged, require_day4_run,
                             snapshot_inputs)
from src.d12_repair_access import RepairedDataAccess, repaired_config
from src.d12_repair_io import require_stage
from src.d12_repo import D12Error, sha256
from src.d12_r1 import participation_summary, simulate_r1


REPAIR_ID = "repair_v3"
R0_RUN_ID = "r0_fixed_v1"
CORE_FRAMES = ("daily", "targets", "orders", "trades", "audits", "predictions",
               "factor_exposures")


def say(message):
    print(message, flush=True)


def project_path(value, top):
    """Resolve a recorded absolute project path without accepting a foreign tree."""
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists() and path.resolve().is_relative_to(ROOT):
        return path.resolve()
    parts = path.parts
    if top not in parts:
        raise D12Error(f"Recorded path has no {top!r} anchor: {value}")
    candidate = ROOT.joinpath(*parts[parts.index(top):]).resolve()
    if not candidate.is_relative_to(ROOT) or not candidate.exists():
        raise D12Error(f"Recorded Day-3 path is unavailable: {candidate}")
    return candidate


def run_preflight(output, config):
    report = preflight.Report()
    baseline = json.loads((ROOT / "config/d11_baseline.json").read_text(encoding="utf-8"))
    for check, rest in [
            (preflight.check_repository, [baseline]),
            (preflight.check_inputs, [config]),
            (preflight.check_registered_digests, []),
            (preflight.check_environment, [config]),
            (preflight.check_saved_runs, [])]:
        check(report, *rest)
    output.json("preflight.json", {"stages": report.stages})
    if report.blocking:
        raise D12Error("Day-4 preflight blocked; see preflight.json")
    return report.stages[0]["detail"]


def validate_frozen_spec(config, experiments):
    r1 = experiments["r1"]
    checks = {
        "r1_frozen": r1.get("frozen") is True,
        "decision_k": float(r1.get("k_decision")) == .5,
        "extra_multiplier": float(r1.get("extra_multiplier")) == 1.,
        "execution_k_nominal": float(r1.get("k_execution_nominal")) == .5,
        "execution_config_matches": float(config["costs"]["impact"]["coefficient"]) == .5,
        "penalty_expression": r1["penalty_expression"]["mode"] == "abs",
        "amount_minimum": int(r1["a_est"]["minimum_observations"]) == 15,
        "amount_unavailable_90pct": "90" in r1["a_est"]["unavailable_policy"],
        "amount_lookback": int(config["tradability"]["amount_lookback_trade_days"]) == 20,
        "volatility_minimum": int(r1["sigma_est"]["minimum_observations"]) == 20,
        "volatility_lookback": int(config["costs"]["impact"]["volatility_lookback_trade_days"]) == 20,
        "fallback": float(r1["sigma_est"]["fallback_daily_volatility"]) == .03,
        "execution_fallback_matches": float(config["costs"]["impact"]["fallback_daily_volatility"]) == .03,
        "turnover_cap": float(config["d7"]["single_sided_turnover_limit"]) == .15,
        "participation_monitoring_only": "不加上限" in r1["implementation_risk"]["participation_cap_asymmetry"],
    }
    if not all(checks.values()):
        raise D12Error("Frozen R1 specification no longer matches the implementation inputs: "
                       + json.dumps(checks, ensure_ascii=False))
    return {"checks": checks, "experiment_registry_sha256": sha256(ROOT / "reports/d12/experiment_registry.tsv"),
            "experiments_sha256": sha256(ROOT / "config/d12_experiments.yaml"),
            "participation_constraint_added": False,
            "comparison_status": "provisional_due_to_unresolved_P0_12"}


def verify_repaired_r0(run):
    manifest_path = run / "repaired_r0_manifest.json"
    if not manifest_path.is_file():
        raise D12Error("Repaired R0 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("classification") != "repaired_r0_after_factor_chain_fix":
        raise D12Error("Repaired R0 is not a completed repair_v3 baseline")
    required = {f"{name}.parquet" for name in CORE_FRAMES}
    if not required.issubset(manifest.get("output_sha256", {})):
        raise D12Error("Repaired R0 manifest does not declare all Day-4 inputs")
    for number, (name, digest) in enumerate(manifest["output_sha256"].items(), 1):
        candidate = run / name
        if Path(name).name != name or not candidate.is_file() or sha256(candidate) != digest:
            raise D12Error(f"Repaired R0 output missing or changed: {name}")
        if number % 100 == 0:
            say(f"D12 Day4 repaired R0 SHA256 {number}/{len(manifest['output_sha256'])}")
    return manifest


def load_handoff():
    stage_manifests = {}
    for stage in ["classify", "provider", "inputs", "r0", "compare"]:
        _, stage_manifests[stage] = require_stage(ROOT, REPAIR_ID, stage)
    inputs = stage_manifests["inputs"]
    provider = project_path(inputs["corrected_provider"], "data")
    design = project_path(inputs["processed_design_dir"], "data")
    oos = project_path(inputs["processed_oos_dir"], "data")
    r0_stage = stage_manifests["r0"]
    run = project_path(r0_stage["run_directory"], "reports")
    if run.name != R0_RUN_ID or run.parts[-3:] != ("r0_fixed", REPAIR_ID, R0_RUN_ID):
        raise D12Error(f"Day 4 must use {REPAIR_ID}/{R0_RUN_ID}, actual={run}")
    manifest = verify_repaired_r0(run)
    if project_path(manifest["corrected_provider"], "data") != provider:
        raise D12Error("Repaired R0 provider identity differs from the repaired-input stage")
    if manifest.get("membership_repair") != "not attempted; P0-12 remains blocked":
        raise D12Error("Unexpected membership status in repaired R0")
    return {"provider": provider, "design": design, "oos": oos, "run": run,
            "r0_manifest": manifest, "stage_manifests": stage_manifests}


def factor_columns(exposures, config):
    industries = [column for column in exposures
                  if column.startswith("industry_")
                  and column not in {"industry_name", "industry_factor"}
                  and pd.api.types.is_numeric_dtype(exposures[column])]
    styles = ["style_" + name for name in config["d7"]["constrained_style_factors"]]
    if not industries or not set(styles).issubset(exposures):
        raise D12Error(f"Cannot recover repaired R0 factor schema: industries={len(industries)}, styles={styles}")
    if "industry_factor" in exposures:
        declared = set(exposures.industry_factor.dropna().astype(str).unique())
        if not declared.issubset(industries):
            raise D12Error("Saved industry labels do not match numeric dummy columns")
    return industries, styles


def load_inputs(config, handoff):
    access = RepairedDataAccess(ROOT, handoff["provider"])
    repaired = repaired_config(config, processed_design_dir=handoff["design"].relative_to(ROOT),
                               processed_oos_dir=handoff["oos"].relative_to(ROOT))
    start, end = pd.Timestamp(repaired["windows"]["design_start"]), pd.Timestamp(repaired["windows"]["oos_end"])
    calendar = load_open_calendar(access, repaired, end_date=end)
    weights, flags = load_weights_flags(access, repaired, start, end)
    say("D12 Day4 loading corrected market; no model fitting")
    market, _ = prepare_oos_market(access, repaired, weights, flags)
    saved = {name: pd.read_parquet(handoff["run"] / f"{name}.parquet") for name in CORE_FRAMES}
    industries, styles = factor_columns(saved["factor_exposures"], repaired)
    return repaired, market, flags, calendar, saved, industries, styles


def daily_result(result, reference, capital):
    daily = result.nav.copy()
    benchmark = reference[["trade_date", "benchmark_return"]].copy()
    daily = daily.merge(benchmark, on="trade_date", how="left", validate="one_to_one")
    daily["portfolio_return"] = daily.nav / daily.nav.shift(1, fill_value=float(capital)) - 1
    if not np.isfinite(daily[["portfolio_return", "benchmark_return"]]).all().all():
        raise D12Error("Day-4 daily result has missing or non-finite returns")
    return daily


def zero_gate(result, daily, saved):
    if not daily.trade_date.equals(saved["daily"].trade_date):
        raise D12Error("Zero gate date mismatch")
    errors = {
        "daily": check_parity(daily, saved["daily"], ["trade_date"],
                              ["nav", "cash", "transaction_cost"], "repaired R0 account", atol=1e-4),
        "returns": check_parity(daily, saved["daily"], ["trade_date"],
                                ["single_sided_turnover", "portfolio_return"], "repaired R0 returns"),
        "targets": check_parity(result.targets, saved["targets"], ["signal_date", "instrument"],
                                ["target_weight", "pretrade_weight"], "repaired R0 targets"),
        "orders": check_parity(result.orders, saved["orders"], ["signal_date", "instrument"],
                               ["order_units", "signal_nav"], "repaired R0 orders", atol=1e-5),
        "trades": check_parity(result.trades, saved["trades"], ["execution_date", "instrument"],
                               ["executed_units", "total_cost"], "repaired R0 fills", atol=1e-5),
    }
    for new, old, keys, columns in [
            (result.orders, saved["orders"], ["signal_date", "instrument"],
             ["execution_date", "signal_can_buy"]),
            (result.trades, saved["trades"], ["execution_date", "instrument"],
             ["exchange_can_buy", "exchange_can_sell", "participation_limited"]),
            (result.audits, saved["audits"], ["signal_date"],
             ["solver_status", "fallback_used", "initial_construction_exempt"])]:
        left, right = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
        if not left.index.equals(right.index) or not left[columns].equals(right[columns]):
            raise D12Error("Zero gate changed a discrete order, fill or solver outcome")
    return {"passed": True, "full_period": True, "days": int(len(daily)),
            "decisions": int(len(result.audits)), "source_baseline": str(Path("reports/d12/r0_fixed") / REPAIR_ID / R0_RUN_ID),
            "not_compared_to": "reports/d11/d11_v1", "decision_k": 0.,
            "structural_bypass": "src.d12_r1.simulate_r1 -> src.d11_core.simulate_causal",
            "maximum_errors": errors, "rtol": 2e-9,
            "atol": {"daily": 1e-4, "returns": 1e-9, "targets": 1e-9,
                     "orders": 1e-5, "trades": 1e-5},
            "r1_zero_coefficient_gate_satisfied": True,
            "p0_membership_quality_certified": False}


def verify_gate(run_id):
    path, manifest = require_day4_run(ROOT, run_id)
    gate_path = path / "zero_gate.json"
    if manifest.get("mode") != "zero_gate" or manifest.get("decision_k") != 0. or not gate_path.is_file():
        raise D12Error("The prerequisite run is not a zero-coefficient gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("source_baseline") != str(Path("reports/d12/r0_fixed") / REPAIR_ID / R0_RUN_ID):
        raise D12Error("Zero gate did not pass against repair_v3/r0_fixed_v1")
    return {"run_id": run_id, "manifest_sha256": sha256(path / "manifest.json"),
            "gate_sha256": sha256(gate_path), "gate": gate}


def save_simulation(output, result, daily):
    for name, frame in [("daily", daily), ("targets", result.targets), ("orders", result.orders),
                        ("trades", result.trades), ("audits", result.audits)]:
        output.parquet(name + ".parquet", frame)


def save_snapshot(output, name, snapshot):
    """Keep the complete inventory locally as ignored Parquet; commit only its identity."""
    serial = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    frame = pd.DataFrame([{"path": path, **identity} for path, identity in snapshot.items()])
    output.parquet(name + ".parquet", frame)
    summary = {"files": int(len(frame)), "bytes": int(frame.bytes.sum()),
               "canonical_json_sha256": hashlib.sha256(serial).hexdigest(),
               "detail_file": name + ".parquet",
               "detail_note": "gitignored local evidence; manifest records its file SHA256"}
    output.json(name + "_summary.json", summary)
    return summary


def flatten_performance(performance):
    flat = {}
    for key, value in performance.items():
        if key == "newey_west_t_by_lag":
            for lag, item in value.items():
                flat[f"newey_west_t_lag_{lag}"] = item
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = value
    return flat


def three_way_table(original, repaired, r1):
    versions = {"original_d11": flatten_performance(original),
                "repaired_r0": flatten_performance(repaired),
                "r1_nominal": flatten_performance(r1)}
    rows = []
    for metric in sorted(set().union(*(value.keys() for value in versions.values()))):
        old, fixed, candidate = (versions[name].get(metric) for name in versions)
        rows.append({"metric": metric, "original_d11": old, "repaired_r0": fixed,
                     "r1_nominal": candidate,
                     "repair_change_not_strategy": fixed - old if old is not None and fixed is not None else None,
                     "r1_minus_repaired_strategy_comparison": candidate - fixed if candidate is not None and fixed is not None else None})
    return pd.DataFrame(rows)


def monthly_comparison(original_daily, repaired_daily, r1_daily, config):
    frames = []
    for label, daily in [("original_d11", original_daily), ("repaired_r0", repaired_daily), ("r1_nominal", r1_daily)]:
        _, monthly = summarize_monthly_performance(daily, newey_west_lags=config["d9"]["newey_west_lags"])
        frames.append(monthly[["month", "excess_return"]].rename(columns={"excess_return": label}))
    result = frames[0].merge(frames[1], on="month", how="outer").merge(frames[2], on="month", how="outer")
    result["repair_change_not_strategy"] = result.repaired_r0 - result.original_d11
    result["r1_minus_repaired"] = result.r1_nominal - result.repaired_r0
    return result.sort_values("month")


def write_report(output, performance, comparison, monitoring):
    values = flatten_performance(performance)
    metric = comparison.set_index("metric")
    delta = metric.r1_minus_repaired_strategy_comparison
    text = f"""# D12 第四天：R1 名义运行结果

状态：已通过修复 R0 的零系数全期门；R1 只新增冻结的事前冲击惩罚，复用修复 R0 预测。

| 指标 | R1 | R1 − 修复 R0 |
|---|---:|---:|
| 年化超额 | {values['annualized_excess_return']:.6%} | {delta['annualized_excess_return']:+.6%} |
| IR | {values['information_ratio']:.6f} | {delta['information_ratio']:+.6f} |
| 年化跟踪误差 | {values['annualized_tracking_error']:.6%} | {delta['annualized_tracking_error']:+.6%} |
| NW-t 最小值 | {values['newey_west_t_minimum']:.6f} | {delta['newey_west_t_minimum']:+.6f} |
| 最大相对回撤 | {values['maximum_relative_drawdown']:.6%} | {delta['maximum_relative_drawdown']:+.6%} |

- 原 D11、修复 R0 与 R1 的完整三列表见 `three_way_results.tsv`；修复差异单列，未记为策略改进。
- 买卖两侧参与率、5% 超限、执行端实际截断与未成交金额见 `participation_summary.tsv`。
- 本日未做归因、IC 或执行端冲击 ×2；它们按排期属于第五天。
- P0-12 成分生效日仍未修复，因此这是同一缺陷口径下的受控比较，不恢复历史 Go/No-Go；历史结论仍为 No-Go。
- 2019–2026 历史已被反复查看，本次不是新的样本外验证。

监测摘要：{monitoring.to_dict(orient='records')}
"""
    output.text("day4_r1_report.md", text)


def run_stage(args, output):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    experiments = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text(encoding="utf-8"))
    repository = run_preflight(output, config)
    frozen = validate_frozen_spec(config, experiments)
    output.json("frozen_spec_identity.json", frozen)
    handoff = load_handoff()
    output.json("repaired_r0_identity.json", {
        "repair_id": REPAIR_ID, "run_id": R0_RUN_ID,
        "provider": str(handoff["provider"].relative_to(ROOT)),
        "processed_design_dir": str(handoff["design"].relative_to(ROOT)),
        "processed_oos_dir": str(handoff["oos"].relative_to(ROOT)),
        "manifest_sha256": sha256(handoff["run"] / "repaired_r0_manifest.json"),
        "outputs_verified": len(handoff["r0_manifest"]["output_sha256"]),
        "performance": handoff["r0_manifest"]["research"]["performance"],
        "membership_status": "P0-12 remains blocked; comparison is provisional"})
    before = snapshot_inputs(ROOT, config)
    before_summary = save_snapshot(output, "inputs_before", before)
    repaired, market, flags, calendar, saved, industries, styles = load_inputs(config, handoff)
    total_decisions = len(saved["audits"])
    count = 0
    def progress(target, orders, audit):
        nonlocal count
        count += 1
        if count == 1 or count % 25 == 0 or count == total_decisions:
            say(f"D12 Day4 {args.stage}: {count}/{total_decisions} decisions")

    if args.stage == "gate":
        estimator = ImpactEstimator.from_config(repaired, k=0.)
        result = simulate_r1(market, saved["predictions"], saved["factor_exposures"],
                             flags, calendar, repaired, industries, styles, estimator, progress=progress)
        daily = daily_result(result, saved["daily"], repaired["d8"]["main_capital_cny"])
        gate = zero_gate(result, daily, saved)
        output.json("zero_gate.json", gate)
        save_simulation(output, result, daily)
        result_summary = {"mode": "zero_gate", "decision_k": 0., "zero_gate": gate,
                          "historical_decision": "No-Go", "new_oos_validation": False}
    else:
        gate_identity = verify_gate(args.gate_run_id)
        output.json("zero_gate_identity.json", gate_identity)
        k = float(experiments["r1"]["k_decision"]) * float(experiments["r1"]["extra_multiplier"])
        estimator = ImpactEstimator.from_config(repaired, k=k)
        result = simulate_r1(market, saved["predictions"], saved["factor_exposures"],
                             flags, calendar, repaired, industries, styles, estimator, progress=progress)
        daily = daily_result(result, saved["daily"], repaired["d8"]["main_capital_cny"])
        performance, _ = summarize_monthly_performance(daily, newey_west_lags=repaired["d9"]["newey_west_lags"])
        monitoring = participation_summary(result.orders, result.trades)
        original_daily = pd.read_parquet(ROOT / "reports/d11/d11_v1/daily.parquet")
        comparison = three_way_table(
            json.loads((ROOT / "reports/d12/repair/repair_v3/compare/comparison.json").read_text())["original_performance"],
            handoff["r0_manifest"]["research"]["performance"], performance)
        monthly = monthly_comparison(original_daily, saved["daily"], daily, repaired)
        output.json("r1_summary.json", {
            "classification": "registered_R1_nominal_on_repaired_R0",
            "decision_k": k, "execution_k": float(repaired["costs"]["impact"]["coefficient"]),
            "extra_multiplier": float(experiments["r1"]["extra_multiplier"]),
            "predictions_reused_from": str(handoff["run"].relative_to(ROOT)),
            "model_fitted": False, "performance": performance,
            "decisions": int(len(result.audits)), "solver_status_counts": result.audits.solver_status.value_counts().to_dict(),
            "fallback_used": int(result.audits.fallback_used.sum()),
            "amount_buy_forbidden_name_decisions": int(result.audits.amount_buy_forbidden_names.sum()),
            "sigma_source_counts": {
                "rolling": int(result.audits.sigma_source_rolling.sum()),
                "cross_section_median": int(result.audits.sigma_source_cross_section_median.sum()),
                "constant_fallback": int(result.audits.sigma_source_constant_fallback.sum())},
            "zero_gate": gate_identity, "historical_decision": "No-Go",
            "comparison_status": "provisional_due_to_unresolved_P0_12",
            "new_oos_validation": False, "day5_pending": ["repaired R0 and R1 attribution/IC", "execution impact x2"]})
        output.table("three_way_results", comparison)
        output.table("monthly_three_way_comparison", monthly)
        output.table("participation_summary", monitoring)
        output.table("decision_telemetry", result.audits)
        write_report(output, performance, comparison, monitoring)
        save_simulation(output, result, daily)
        result_summary = {"mode": "r1_nominal", "decision_k": k, "performance": performance,
                          "zero_gate_run_id": args.gate_run_id, "historical_decision": "No-Go",
                          "new_oos_validation": False}

    after = snapshot_inputs(ROOT, config)
    after_summary = save_snapshot(output, "inputs_after", after)
    assert_snapshot_unchanged(before, after)
    if before_summary["canonical_json_sha256"] != after_summary["canonical_json_sha256"]:
        raise D12Error("Day-4 input snapshot digest changed")
    return {**result_summary, "repository": repository, "inputs_unchanged": True,
            "repaired_r0": str(handoff["run"].relative_to(ROOT)),
            "unresolved": ["P0-12 membership timing", "P0-24 alias missing prices", "P0-27 excluded factor breaks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["gate", "r1"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gate-run-id", help="required for --stage r1; completed Day-4 zero gate")
    args = parser.parse_args()
    if args.stage == "r1" and not args.gate_run_id:
        parser.error("--stage r1 requires --gate-run-id")
    if args.stage == "gate" and args.gate_run_id:
        parser.error("--gate-run-id is only valid with --stage r1")
    output = Day4Output(ROOT, args.run_id)
    started = time.perf_counter()
    manifest = {"run_id": args.run_id, "stage": args.stage, "status": "running",
                "inputs_unchanged": False}
    try:
        result = run_stage(args, output)
        manifest.update(result, status="completed")
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        output.text("failure.txt", traceback.format_exc())
        say(f"{type(error).__name__}: {error}")
    manifest.update(recorded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    elapsed_seconds=time.perf_counter() - started, output_sha256=output.hashes())
    output.json("manifest.json", manifest)
    say(f"Wrote {output.path}; status={manifest['status']}; stage={args.stage}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

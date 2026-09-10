"""Day-6 diagnostic/freeze tests; no Qlib provider is required."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_day6 import (constraint_tradability_review, evidence_matrix,
                          feature_distribution_review, feature_extreme_observations,
                          forward_startup_checklist,
                          monthly_concentration, r2_readiness,
                          signal_year_diagnostics, validate_day6_freeze)
from src.d12_day6_io import Day6Output, require_day5_run, snapshot_inputs
from src.d12_repo import D12Error, sha256


CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
EXPERIMENTS = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text(
    encoding="utf-8"))


def signal_fixture():
    predictions, labels = [], []
    for year in [2023, 2025]:
        for week in [1, 2]:
            date = pd.Timestamp(year=year, month=1, day=week * 7)
            for number in range(20):
                score = float(number)
                if year == 2023:
                    realized = float((number * 7) % 19 - 9) / 1000
                else:
                    realized = number / 1000
                    if number < 2:
                        realized = -.005 + number / 1000
                    if number >= 18:
                        realized = -.020 + (number - 18) / 1000
                predictions.append({"sample_date": date, "instrument": f"S{number:02d}",
                                    "prediction": score,
                                    "expected_5d_excess": score / 2000,
                                    "feature_available": True})
                labels.append({"sample_date": date, "instrument": f"S{number:02d}",
                               "excess_return": realized,
                               "label_end_timestamp": date + pd.Timedelta(days=7)})
    return pd.DataFrame(predictions), pd.DataFrame(labels)


def test_day6_run_ids_are_immutable_and_actionable(tmp_path):
    Day6Output(tmp_path, "day6_v1")
    with pytest.raises(D12Error, match="already exists and is immutable.*new --run-id"):
        Day6Output(tmp_path, "day6_v1")


def test_day6_snapshot_includes_day5_and_excludes_own_output(tmp_path):
    (tmp_path / "reports/d12/day5/day5_v1").mkdir(parents=True)
    (tmp_path / "reports/d12/day5/day5_v1/manifest.json").write_text("day5")
    (tmp_path / "reports/d12/day6/day6_v1").mkdir(parents=True)
    (tmp_path / "reports/d12/day6/day6_v1/partial.json").write_text("day6")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/input.txt").write_text("input")
    provider = tmp_path / "provider"
    (provider / "features").mkdir(parents=True)
    (provider / "features/a.bin").write_text("provider")
    result = snapshot_inputs(tmp_path, {"data": {"qlib_provider_uri": str(provider)}})
    assert "reports/d12/day5/day5_v1/manifest.json" in result
    assert "data/input.txt" in result
    assert not any(name.startswith("reports/d12/day6/") for name in result)


def test_require_day5_checks_every_registered_local_output(tmp_path):
    path = tmp_path / "reports/d12/day5/day5_v1"
    path.mkdir(parents=True)
    required = ["summary.json", "performance_matrix.tsv", "yearly_ic.tsv",
                "signal_calibration_deciles.tsv", "reference_gates.tsv",
                "repaired_r0_nominal_monthly_performance.parquet",
                "r1_nominal_monthly_performance.parquet"]
    for name in required:
        (path / name).write_text(name)
    outputs = {name: sha256(path / name) for name in required}
    (path / "manifest.json").write_text(json.dumps({
        "status": "completed", "inputs_unchanged": True,
        "mode": "day5_diagnostics_and_stress", "output_sha256": outputs}))
    _, manifest = require_day5_run(tmp_path, "day5_v1")
    assert manifest["status"] == "completed"
    (path / "yearly_ic.tsv").write_text("changed")
    with pytest.raises(D12Error, match="output missing or changed"):
        require_day5_run(tmp_path, "day5_v1")


def test_signal_diagnostic_keeps_fixed_years_and_exposes_2025_tail_failure():
    predictions, labels = signal_fixture()
    deciles, by_date, yearly, summary = signal_year_diagnostics(
        predictions, labels, "2026-08-31")
    assert set(yearly.year) == {2023, 2025}
    assert len(deciles) == 20 and len(by_date) == 4
    result = yearly.set_index("year")
    assert result.loc[2025, "rank_ic_mean"] > 0
    assert result.loc[2025, "top_minus_bottom_mean"] < 0
    assert result.loc[2025, "registered_issue"] == "positive_rank_ic_negative_tail_spread"
    assert summary["model_or_cutpoints_changed"] is False
    with pytest.raises(Exception, match="frozen"):
        signal_year_diagnostics(predictions, labels, "2026-08-31", years=(2025,))


def test_feature_review_uses_only_strictly_earlier_years_and_flags_shift():
    rows = []
    for year, level in [(2019, 0.), (2020, 0.), (2023, 5.), (2025, 10.)]:
        for number in range(20):
            rows.append({"sample_date": pd.Timestamp(year, 1, 5),
                         "instrument": f"S{number:02d}",
                         "stable": number / 10, "shifted": number / 10 + level})
    detail, summary = feature_distribution_review(
        pd.DataFrame(rows), ["stable", "shifted"])
    shifted = detail.loc[detail.feature.eq("shifted")].set_index("year")
    assert shifted.loc[2023, "reference_rows"] == 40
    assert shifted.loc[2025, "reference_rows"] == 60
    assert shifted.review_flag.all()
    assert summary.set_index("year").loc[2025, "review_flags"] >= 1
    assert "not an automatic data-error" in summary.iloc[0].interpretation
    extremes = feature_extreme_observations(pd.DataFrame(rows), detail, top_n=2)
    assert set(extremes.year) == {2023, 2025}
    assert extremes.groupby(["year", "feature"]).size().le(2).all()
    assert extremes.interpretation.str.contains("not a data-error verdict").all()


def path_fixture():
    target_rows, audit_rows, trade_rows = [], [], []
    for year in [2023, 2025]:
        date = pd.Timestamp(year, 1, 6)
        for number in range(10):
            target_rows.append({"signal_date": date, "instrument": f"S{number}",
                                "signal_can_buy": number != 0})
        audit_rows.append({"signal_date": date, "prohibited_or_unknown_buy_names": 1,
                           "target_active_turnover": .15,
                           "maximum_stock_active": .015,
                           "maximum_industry_active": .01,
                           "maximum_style_active": .30,
                           "mandatory_turnover": .001,
                           "prohibited_buy_weight": 0.})
        trade_rows.append({"execution_date": date + pd.Timedelta(days=3),
                           "participation_limited": True, "unfilled_units": 2.,
                           "signal_price": 10., "side": "buy", "cash_scale": .9,
                           "executed_units": 8.})
    return pd.DataFrame(target_rows), pd.DataFrame(audit_rows), pd.DataFrame(trade_rows)


def test_constraint_review_is_observed_telemetry_not_counterfactual():
    result = constraint_tradability_review(*path_fixture(), CONFIG, "r1")
    assert result.turnover_cap_binding_fraction.eq(1).all()
    assert result.style_limit_binding_fraction.eq(1).all()
    assert result.industry_limit_binding_fraction.eq(0).all()
    assert result.unfilled_signal_cny.eq(20).all()
    assert result.interpretation.str.contains("not a constraint-relaxation").all()


def test_month_concentration_preserves_three_baseline_identities():
    monthly = pd.DataFrame({"month": ["2025-01", "2025-02", "2025-03"],
                            "original_d11": [.01, .02, -.01],
                            "repaired_r0": [.005, .01, -.02],
                            "r1_nominal": [.004, .012, -.02]})
    result = monthly_concentration(monthly)
    assert result.strategy.tolist() == ["original_d11", "repaired_r0", "r1_nominal"]
    assert result.best_five_share_of_positive_excess.eq(1).all()


def test_day6_freeze_resolves_fold_count_and_keeps_forward_honest():
    checks = validate_day6_freeze(EXPERIMENTS)
    assert all(checks.values())
    checklist = forward_startup_checklist(EXPERIMENTS)
    assert checklist.loc[checklist.item.eq("point-in-time membership evidence"),
                         "status"].item() == "blocked"
    assert checklist.loc[checklist.item.eq("forward logging start"),
                         "evidence_or_requirement"].item() == "2026-09-21"


def test_r2_readiness_stays_blocked_and_evidence_matrix_marks_r2_unrun():
    table, decision = r2_readiness(
        ["P0-12"], day5_completed=True,
        p2_manual_review_completed=False, resource_budget_verified=False)
    assert decision["decision"] == "not_started_blocked"
    assert decision["r2_run_this_week"] is False
    assert set(table.status) == {"passed", "blocked", "pending_manual_review", "unverified"}
    metrics = {"annualized_excess_return": .04, "annualized_tracking_error": .06,
               "information_ratio": .7, "newey_west_t_minimum": 1.8,
               "maximum_relative_drawdown": .11}
    summary = {"performance": {name: metrics for name in [
        "original_d11_reference", "repaired_r0_nominal", "r1_nominal",
        "repaired_r0_impact2x", "r1_impact2x"]}}
    result = evidence_matrix(summary)
    assert result.loc[result.scenario.eq("r2"), "status"].item() == "not_run"

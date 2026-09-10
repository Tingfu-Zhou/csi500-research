"""Day-5 diagnostics/stress wiring tests; no Qlib provider is required."""
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_day5 import (attribution_comparison, daily_from_simulation,
                          impact2x_configuration, nominal_replay_parity,
                          performance_matrix, reference_gates, signal_calibration,
                          stress_effects)
from src.d12_day5_io import Day5Output, snapshot_inputs
from src.d12_repo import D12Error


CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text())


def test_day5_run_ids_are_immutable_and_error_is_actionable(tmp_path):
    Day5Output(tmp_path, "day5_v1")
    with pytest.raises(D12Error, match="already exists and is immutable.*new --run-id"):
        Day5Output(tmp_path, "day5_v1")


def test_day5_snapshot_includes_day4_but_excludes_own_outputs(tmp_path):
    (tmp_path / "reports/d12/day4/r1").mkdir(parents=True)
    (tmp_path / "reports/d12/day4/r1/manifest.json").write_text("day4")
    (tmp_path / "reports/d12/day5/run").mkdir(parents=True)
    (tmp_path / "reports/d12/day5/run/partial.json").write_text("day5")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/input.txt").write_text("input")
    provider = tmp_path / "provider"
    (provider / "features").mkdir(parents=True)
    (provider / "features/a.bin").write_text("provider")
    config = {"data": {"qlib_provider_uri": str(provider)}}
    result = snapshot_inputs(tmp_path, config)
    assert "reports/d12/day4/r1/manifest.json" in result
    assert "data/input.txt" in result
    assert not any(name.startswith("reports/d12/day5/") for name in result)


def test_impact2x_changes_execution_only_and_does_not_mutate_config():
    stressed, decision = impact2x_configuration(CONFIG)
    assert CONFIG["costs"]["impact"]["coefficient"] == .5
    assert stressed["costs"]["impact"]["coefficient"] == 1.
    assert decision.k == .5
    changed = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    changed["costs"]["impact"]["coefficient"] = .6
    with pytest.raises(D12Error, match="nominal execution impact"):
        impact2x_configuration(changed)


def test_daily_from_simulation_uses_actual_nav_feedback():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    result = SimpleNamespace(nav=pd.DataFrame({"trade_date": dates,
                                                "nav": [100., 101.], "cash": [10., 9.]}))
    benchmark = pd.DataFrame({"trade_date": dates, "benchmark_return": [0., .005]})
    daily = daily_from_simulation(result, benchmark, 100.)
    assert daily.portfolio_return.tolist() == pytest.approx([0., .01])


def parity_fixture():
    d0, d1 = pd.to_datetime(["2025-01-02", "2025-01-03"])
    daily = pd.DataFrame({"trade_date": [d0, d1], "nav": [100., 101.],
                          "cash": [0., 0.], "transaction_cost": [0., .1],
                          "single_sided_turnover": [0., .1],
                          "portfolio_return": [0., .01], "benchmark_return": [0., .005]})
    targets = pd.DataFrame({"signal_date": [d0], "instrument": ["A"],
                            "target_weight": [1.], "pretrade_weight": [0.]})
    orders = pd.DataFrame({"signal_date": [d0], "execution_date": [d1],
                           "instrument": ["A"], "order_units": [10.], "signal_nav": [100.],
                           "decision_coefficient": [.005], "expected_impact_fraction": [.001],
                           "signal_can_buy": [True], "amount_buy_forbidden": [False],
                           "sigma_source": ["rolling"]})
    trades = orders.assign(executed_units=10., total_cost=.1, exchange_can_buy=True,
                           exchange_can_sell=True, participation_limited=False)
    audits = pd.DataFrame({"signal_date": [d0], "target_active_turnover": [0.],
                           "expected_impact_fraction": [.001], "decision_impact_k": [.5],
                           "solver_status": ["optimal"], "fallback_used": [False],
                           "initial_construction_exempt": [True]})
    result = SimpleNamespace(targets=targets.copy(), orders=orders.copy(),
                             trades=trades.copy(), audits=audits.copy())
    saved = {"daily": daily, "targets": targets, "orders": orders,
             "trades": trades, "audits": audits}
    return result, daily.copy(), saved


def test_nominal_r1_replay_requires_all_numeric_and_discrete_states():
    result, daily, saved = parity_fixture()
    check = nominal_replay_parity(result, daily, saved)
    assert check["passed"] and set(check["maximum_errors"]) == {
        "daily", "returns", "targets", "orders", "order_impact", "trades", "audits"}
    result.orders.loc[0, "expected_impact_fraction"] += 1e-8
    with pytest.raises(Exception, match="order impact"):
        nominal_replay_parity(result, daily, saved)
    result.orders.loc[0, "expected_impact_fraction"] -= 1e-8
    result.orders.loc[0, "sigma_source"] = "constant_fallback"
    with pytest.raises(Exception, match="discrete decision"):
        nominal_replay_parity(result, daily, saved)


def test_signal_calibration_is_date_balanced_into_ten_deciles():
    rows, labels = [], []
    for date in pd.to_datetime(["2025-01-03", "2025-01-10"]):
        for number in range(20):
            rows.append({"sample_date": date, "instrument": f"S{number:02d}",
                         "prediction": float(number),
                         "expected_5d_excess": number / 1000,
                         "feature_available": True})
            labels.append({"sample_date": date, "instrument": f"S{number:02d}",
                           "excess_return": number / 500,
                           "label_end_timestamp": date + pd.Timedelta(days=7)})
    table, summary = signal_calibration(pd.DataFrame(rows), pd.DataFrame(labels),
                                        "2025-12-31", deciles=10)
    assert table.decile.tolist() == list(range(1, 11))
    assert table.observations.eq(4).all()
    assert summary["adjacent_up_steps"] == 9
    assert summary["top_minus_bottom_realized"] > 0


def scenario(performance):
    return {"decision_k": .5, "execution_k": .5,
            "performance": performance,
            "performance_detail": {"same_path_gross": {
                "annualized_excess_return": performance["annualized_excess_return"] + .01},
                "cost_drag_annualized": .01},
            "execution": {"total_cost_cny": 10., "costs_cny": {"impact_cost": 4.}}}


def test_performance_matrix_and_stress_effects_keep_questions_separate():
    base = {"annualized_excess_return": .04, "annualized_tracking_error": .06,
            "information_ratio": 2 / 3, "newey_west_t_minimum": 1.8,
            "maximum_relative_drawdown": .11}
    scenarios = {
        "repaired_r0_nominal": scenario(base),
        "r1_nominal": scenario({**base, "annualized_excess_return": .041}),
        "repaired_r0_impact2x": scenario({**base, "annualized_excess_return": .03}),
        "r1_impact2x": scenario({**base, "annualized_excess_return": .032}),
    }
    matrix = performance_matrix(scenarios)
    effects = stress_effects(matrix).set_index("comparison")
    assert effects.loc["r1_nominal_minus_r0_nominal",
                       "annualized_excess_return_difference"] == pytest.approx(.001)
    assert effects.loc["r1_minus_r0_under_impact2x",
                       "annualized_excess_return_difference"] == pytest.approx(.002)


def attribution(passed, share):
    return {"annualized_contributions": {"industry": .001, "excess_return": .04},
            "specific_share_net": share, "specific_share_gross": share + .1,
            "annualized_mean_absolute_three_block_residual": .004,
            "maximum_calendar_year_annualized_absolute_residual": .006,
            "maximum_stale_exposure_l1": .01,
            "three_block_residual_passed": passed}


def test_attribution_and_six_reference_gates_remain_strategy_specific():
    attrs = {"repaired_r0": attribution(False, .6), "r1": attribution(True, .4)}
    table = attribution_comparison(attrs)
    assert len(table.loc[table.measure.eq("specific_share_net")]) == 2
    nominal = {
        "repaired_r0": {"annualized_excess_return": .04, "information_ratio": .8,
                        "newey_west_t_minimum": 2.1, "maximum_relative_drawdown": .09},
        "r1": {"annualized_excess_return": .03, "information_ratio": .7,
               "newey_west_t_minimum": 1.9, "maximum_relative_drawdown": .11}}
    stress = {"repaired_r0": {"annualized_excess_return": .01},
              "r1": {"annualized_excess_return": -.01}}
    gates = reference_gates(nominal, stress, attrs, CONFIG)
    assert gates.groupby("strategy").size().to_dict() == {"repaired_r0": 6, "r1": 6}
    assert gates.loc[gates.strategy.eq("repaired_r0"), "passed"].sum() == 6
    assert gates.loc[gates.strategy.eq("r1"), "passed"].sum() == 0

"""Day-4 R1 wiring tests; no Qlib provider or saved Parquet is required."""
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d12_day4 import validate_frozen_spec, zero_gate
from src.d3_core import CostModel
from src.d7_core import OptimizationLimits
from src.d12_core import ImpactEstimator, expected_impact_fraction
from src.d12_r1 import (annotate_orders, optimize_at_signal_r1,
                        participation_summary, simulate_r1)


CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text())
EXPERIMENTS = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text())


def optimizer_fixture(names=40, seed=23):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"SZ{i:06d}" for i in range(names)], name="instrument")
    industries = [f"industry_{i}" for i in range(4)]
    exposures = pd.DataFrame(0., index=index, columns=[*industries, "style_Size", "style_Beta"])
    for number, name in enumerate(index):
        exposures.loc[name, industries[number % 4]] = 1.
    exposures["style_Size"] = rng.normal(size=names)
    exposures["style_Beta"] = rng.normal(size=names)
    benchmark = pd.Series(1 / names, index=index)
    return {"mu": pd.Series(rng.normal(0, .01, size=names), index=index),
            "benchmark": benchmark, "pretrade": benchmark.copy(), "exposures": exposures,
            "buy_allowed": pd.Series(True, index=index), "industries": industries,
            "styles": ["style_Size", "style_Beta"],
            "limits": OptimizationLimits(.015, .02, .3, .15, 1e-6),
            "costs": CostModel.from_config(CONFIG), "execution_date": pd.Timestamp("2025-03-03")}


def test_zero_simulator_structurally_delegates_without_estimating(monkeypatch):
    sentinel = SimpleNamespace(nav=pd.DataFrame({"x": [1]}), targets=pd.DataFrame(),
                               orders=pd.DataFrame(), trades=pd.DataFrame(), audits=pd.DataFrame())
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    monkeypatch.setattr("src.d12_r1.simulate_causal", original)
    result = simulate_r1(None, None, None, None, None, None, None, None, ImpactEstimator(k=0.))
    assert len(calls) == 1
    assert result.nav is sentinel.nav and result.targets is sentinel.targets


def test_zero_optimizer_structurally_delegates(monkeypatch):
    setup = optimizer_fixture(8)
    sentinel = (pd.Series(dtype=float), {"passed": True})
    monkeypatch.setattr("src.d12_r1.optimize_at_signal", lambda *args, **kwargs: sentinel)
    result = optimize_at_signal_r1(**setup, coefficients=pd.Series(0., index=setup["benchmark"].index),
                                   amount_buy_forbidden=pd.Series(False, index=setup["benchmark"].index),
                                   estimator=ImpactEstimator(k=0.))
    assert result is sentinel


def test_nonzero_optimizer_keeps_constraints_and_accounts_for_impact():
    setup = optimizer_fixture()
    coefficient = pd.Series(.006, index=setup["benchmark"].index)
    blocked = pd.Series(False, index=setup["benchmark"].index)
    # The highest-alpha name is missing A_est: it may be held, but not increased.
    name = setup["mu"].idxmax()
    blocked.loc[name] = True
    target, audit = optimize_at_signal_r1(
        **setup, coefficients=coefficient, amount_buy_forbidden=blocked,
        estimator=ImpactEstimator(k=.5))
    assert audit["passed"] and audit["solver_status"] == "optimal"
    assert target.loc[name] <= setup["pretrade"].loc[name] + 1e-8
    assert audit["amount_buy_forbidden_names"] == 1
    assert audit["expected_impact_fraction"] > 0
    assert audit["expected_impact_fraction"] == pytest.approx(
        expected_impact_fraction(target - setup["pretrade"], coefficient), rel=1e-8)
    assert audit["target_active_turnover"] <= .15 + 1e-6


def test_forced_exit_is_charged_in_the_decision_objective():
    setup = optimizer_fixture()
    outside = pd.Series({"SZ999999": .01})
    setup["pretrade"] = pd.concat([setup["pretrade"] * .99, outside])
    coefficient = pd.Series(.006, index=setup["pretrade"].index)
    blocked = pd.Series(False, index=setup["pretrade"].index)
    _, audit = optimize_at_signal_r1(
        **setup, coefficients=coefficient, amount_buy_forbidden=blocked,
        estimator=ImpactEstimator(k=.5))
    assert audit["mandatory_turnover"] == pytest.approx(.01)
    assert audit["expected_impact_forced_fraction"] == pytest.approx(.006 * .01 ** 1.5)
    assert audit["expected_impact_fraction"] == pytest.approx(
        audit["expected_impact_variable_fraction"] + audit["expected_impact_forced_fraction"], rel=1e-8)


def test_order_annotation_records_both_sides_without_adding_a_cap():
    date = pd.Timestamp("2025-03-07")
    orders = pd.DataFrame({
        "instrument": ["SZA", "SZB"], "signal_date": date,
        "execution_date": pd.Timestamp("2025-03-10"), "signal_price": [10., 20.],
        "signal_nav": [1e7, 1e7], "signal_units": [0., 1000.],
        "target_weight": [.1, 0.], "order_units": [100000., -1000.],
        "signal_can_buy": [True, True]})
    estimates = pd.DataFrame({"trade_date": date, "instrument": ["SZA", "SZB"],
                              "sigma_est": [.02, .03], "sigma_source": ["rolling", "rolling"],
                              "a_est": [1e7, 2e8], "a_observations": [20, 20]})
    coefficients = pd.Series({"SZA": .01, "SZB": .02})
    annotated = annotate_orders(orders, estimates, date, coefficients,
                                pd.Series(False, index=coefficients.index))
    assert annotated.set_index("instrument").loc["SZA", "ex_ante_participation"] == pytest.approx(.1)
    assert annotated.set_index("instrument").loc["SZA", "ex_ante_above_limit"]
    # Monitoring does not clip or rewrite the submitted quantity.
    assert annotated.set_index("instrument").loc["SZA", "order_units"] == 100000.
    assert set(annotated.side) == {"buy", "sell"}


def test_order_annotation_refuses_a_buy_with_missing_amount():
    date = pd.Timestamp("2025-03-07")
    orders = pd.DataFrame({"instrument": ["SZA"], "signal_date": date,
                           "execution_date": date + pd.Timedelta(days=3), "signal_price": [10.],
                           "signal_nav": [1e7], "signal_units": [0.], "target_weight": [.01],
                           "order_units": [10000.], "signal_can_buy": [True]})
    estimates = pd.DataFrame({"trade_date": [date], "instrument": ["SZA"],
                              "sigma_est": [.02], "sigma_source": ["rolling"],
                              "a_est": [np.nan], "a_observations": [0]})
    with pytest.raises(Exception, match="amount estimate"):
        annotate_orders(orders, estimates, date, pd.Series({"SZA": .02}), pd.Series({"SZA": True}))


def test_participation_summary_separates_decision_and_execution_monitoring():
    orders = pd.DataFrame({"side": ["buy", "sell"], "order_units": [10., -8.],
                           "signal_price": [10., 10.], "ex_ante_above_limit": [True, False],
                           "ex_ante_participation": [.08, .01], "amount_estimate_cny": [1250., 8000.],
                           "expected_impact_fraction": [.001, .002]})
    trades = orders.assign(participation_limited=[True, False], unfilled_units=[2., 0.],
                           executed_units=[8., -8.], cash_scale=[1., 1.])
    summary = participation_summary(orders, trades).set_index("side")
    assert summary.loc["all", "ex_ante_above_5pct_rows"] == 1
    assert summary.loc["all", "actual_capacity_limited_rows"] == 1
    assert summary.loc["buy", "capacity_limited_unfilled_signal_cny"] == 20.


def test_zero_gate_accepts_exact_repaired_r0_and_labels_it_correctly():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    daily = pd.DataFrame({"trade_date": dates, "nav": [100., 101.], "cash": [0., 0.],
                          "transaction_cost": [0., 1.], "single_sided_turnover": [0., .1],
                          "portfolio_return": [0., .01], "benchmark_return": [0., .005]})
    targets = pd.DataFrame({"signal_date": [dates[0]], "instrument": ["SZA"],
                            "target_weight": [1.], "pretrade_weight": [0.]})
    orders = pd.DataFrame({"signal_date": [dates[0]], "execution_date": [dates[1]],
                           "instrument": ["SZA"], "order_units": [10.], "signal_nav": [100.],
                           "signal_can_buy": [True]})
    trades = pd.DataFrame({"signal_date": [dates[0]], "execution_date": [dates[1]],
                           "instrument": ["SZA"], "executed_units": [10.], "total_cost": [1.],
                           "exchange_can_buy": [True], "exchange_can_sell": [True],
                           "participation_limited": [False]})
    audits = pd.DataFrame({"signal_date": [dates[0]], "solver_status": ["optimal"],
                           "fallback_used": [False], "initial_construction_exempt": [True]})
    result = SimpleNamespace(targets=targets.copy(), orders=orders.copy(), trades=trades.copy(), audits=audits.copy())
    gate = zero_gate(result, daily.copy(), {"daily": daily.copy(), "targets": targets,
                                           "orders": orders, "trades": trades, "audits": audits})
    assert gate["passed"] and gate["r1_zero_coefficient_gate_satisfied"]
    assert "r0_fixed/repair_v3/r0_fixed_v1" in gate["source_baseline"]
    assert gate["not_compared_to"] == "reports/d11/d11_v1"


def test_frozen_spec_is_enforced():
    identity = validate_frozen_spec(CONFIG, EXPERIMENTS)
    assert all(identity["checks"].values())
    changed = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text())
    changed["r1"]["k_decision"] = .6
    with pytest.raises(Exception, match="Frozen R1 specification"):
        validate_frozen_spec(CONFIG, changed)
    changed = yaml.safe_load((ROOT / "config/d12_experiments.yaml").read_text())
    changed["r1"]["penalty_expression"]["mode"] = "split"
    with pytest.raises(Exception, match="Frozen R1 specification"):
        validate_frozen_spec(CONFIG, changed)


def r1_simulation_fixture():
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    calendar = pd.bdate_range("2019-11-25", periods=32)
    config["windows"].update(oos_start=str(calendar[22].date()), oos_end=str(calendar[28].date()))
    config["d7"].update(stock_active_limit=.5, industry_active_limit=1.,
                        style_active_limit=1., single_sided_turnover_limit=1.)
    config["d8"]["main_capital_cny"] = 1e7
    rows = []
    for number, date in enumerate(calendar):
        rows.extend([(date, "A", 10 * (1.001 ** number)),
                     (date, "B", 10 * (.9995 ** number))])
    market = pd.DataFrame(rows, columns=["trade_date", "instrument", "close"])
    market = market.assign(vwap=market.close, daily_amount_cny=2e8, volatility=.02,
                           exchange_can_buy=True, exchange_can_sell=True)
    flags = market[["trade_date", "instrument"]].assign(
        is_st=False, is_listed_under_120d=False, is_below_amount_floor=False)
    signals = calendar[[22, 26]]
    predictions = market.loc[market.trade_date.isin(signals), ["trade_date", "instrument"]].rename(
        columns={"trade_date": "sample_date"})
    predictions["expected_5d_excess"] = [.02, 0., 0., .02]
    exposures = predictions.rename(columns={"sample_date": "trade_date"}).assign(benchmark_weight=.5)
    return market, predictions, exposures, flags, calendar, config


def test_full_nonzero_simulator_emits_registered_telemetry_and_ignores_future_at_t():
    market, predictions, exposures, flags, calendar, config = r1_simulation_fixture()
    estimator = ImpactEstimator.from_config(config, k=.5)
    first_predictions = predictions.loc[predictions.sample_date.eq(predictions.sample_date.min())]
    base = simulate_r1(market, first_predictions, exposures, flags, calendar, config, [], [], estimator)
    assert len(base.audits) == 1
    assert base.audits.solver_status.eq("optimal").all()
    assert {"ex_ante_participation", "ex_ante_above_limit", "decision_coefficient"}.issubset(base.orders)
    assert base.audits.expected_impact_fraction.gt(0).all()

    changed = market.copy()
    later = changed.trade_date > first_predictions.sample_date.min()
    changed.loc[later, "close"] *= 3
    changed.loc[later, "daily_amount_cny"] = 1e6
    other = simulate_r1(changed, first_predictions, exposures, flags, calendar, config, [], [], estimator)
    pd.testing.assert_frame_equal(base.targets.reset_index(drop=True), other.targets.reset_index(drop=True))
    pd.testing.assert_frame_equal(base.orders.reset_index(drop=True), other.orders.reset_index(drop=True))

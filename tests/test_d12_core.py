"""D12 day-1 checks: the R1 estimate must equal D11's charged cost and read no future.

These tests do not run a backtest and need no Qlib data. They pin the four things that
have to be true before R1 may be implemented at all: the decision-side formula is the
execution-side one, k=0 leaves R0 untouched, the estimates at T ignore everything after
T, and the penalty is convex enough for the registered solver.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d3_core import CostModel, trade_cost
from src.d7_core import OptimizationLimits
from src.d11_core import optimize_at_signal
from src.d12_core import (ImpactEstimator, coefficients_at, expected_impact_cny,
                          expected_impact_fraction, impact_fraction_of_trade,
                          impact_penalty, rolling_estimates)
from src.d12_repo import D12Error, required_inputs, verify_repository

CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text())
EXECUTION_DATE = pd.Timestamp("2024-03-05")


def cost_model(**overrides):
    model = CostModel.from_config(CONFIG)
    return CostModel(**{**model.__dict__, **overrides})


def market_panel(days=60, names=6, seed=12, *, amount=4e8):
    """Balanced panel; balanced so the cross-sectional fallback tier is comparable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-06-01", periods=days)
    rows = []
    for index in range(names):
        close = 10 * np.cumprod(1 + rng.normal(0, 0.02, size=days))
        for date, price, turnover in zip(dates, close, rng.uniform(0.5, 1.5, days) * amount):
            rows.append((date, f"{index:06d}.SZ", price, turnover))
    return pd.DataFrame(rows, columns=["trade_date", "instrument", "close", "daily_amount_cny"])


# --- the decision-side term is the execution-side term ------------------------------

@pytest.mark.parametrize("k", [0.5, 1.0])
def test_expected_impact_equals_trade_cost_impact(k):
    """(*) k*sigma*sqrt(NAV/A)*|dw|^1.5 is trade_cost's impact term divided by NAV."""
    rng = np.random.default_rng(20260908)
    model = cost_model(impact_coefficient=k)
    for _ in range(400):
        nav = float(rng.uniform(1e6, 5e8))
        sigma, amount = float(rng.uniform(0, 0.12)), float(rng.uniform(1e7, 5e9))
        # Stay inside the 5% participation cap trade_cost enforces; see the asymmetry test.
        ceiling = min(0.02, model.participation_limit * amount / nav)
        weight_change = float(rng.uniform(-ceiling, ceiling))
        charged = trade_cost(weight_change * nav, sigma, amount, EXECUTION_DATE, model)
        estimated = impact_fraction_of_trade(weight_change, sigma, amount, nav, k)
        assert estimated * nav == pytest.approx(charged["impact_cost"], rel=1e-12, abs=1e-9)


def test_expected_impact_covers_buys_sells_and_untraded_names():
    nav, k = 1e7, 0.5
    trades = pd.Series({"a": 0.01, "b": -0.008, "c": 0.0})
    sigma = pd.Series({"a": 0.02, "b": 0.05, "c": 0.03})
    amount = pd.Series({"a": 3e8, "b": 9e7, "c": 1e9})
    coefficient = k * sigma * np.sqrt(nav / amount)
    model = cost_model(impact_coefficient=k)
    charged = sum(trade_cost(trades[name] * nav, sigma[name], amount[name], EXECUTION_DATE,
                             model)["impact_cost"] for name in trades.index)
    assert expected_impact_cny(trades, coefficient, nav) == pytest.approx(charged, rel=1e-12)
    # A name that does not trade contributes nothing, whatever its coefficient.
    assert expected_impact_fraction(trades.drop("c"), coefficient) == pytest.approx(
        expected_impact_fraction(trades, coefficient), rel=1e-12)


def test_r0_objective_carries_no_impact_term_so_r1_cannot_double_count():
    """optimize_at_signal charges commission, transfer and stamp only.

    Changing the impact coefficient alone leaves its target identical, which is what makes
    the R1 penalty a new term rather than a second helping of a cost already deducted.
    """
    setup = optimizer_fixture()
    targets = [optimize_at_signal(**{**setup, "costs": cost_model(impact_coefficient=value)})[0]
               for value in (0.0, 0.5, 5.0)]
    for other in targets[1:]:
        pd.testing.assert_series_equal(targets[0], other)


# --- k = 0 is R0, structurally ------------------------------------------------------

def test_zero_k_bypasses_the_penalty_and_its_missing_data_rules():
    estimator = ImpactEstimator(k=0.0)
    assert estimator.bypassed
    estimates = rolling_estimates(market_panel(), estimator)
    asof = estimates["trade_date"].max()
    # An instrument with no usable turnover estimate would normally be barred from buying.
    estimates.loc[estimates["trade_date"].eq(asof), "a_est"] = np.nan
    coefficient, buy_forbidden, diagnostics = coefficients_at(estimates, asof, 1e7, estimator)
    assert (coefficient == 0).all()
    assert not buy_forbidden.any()
    assert diagnostics["bypassed_for_zero_k"] is True
    assert expected_impact_fraction(pd.Series(0.01, index=coefficient.index), coefficient) == 0.0


def test_nonzero_k_keeps_the_missing_turnover_rule():
    estimator = ImpactEstimator(k=0.5)
    estimates = rolling_estimates(market_panel(), estimator)
    asof = estimates["trade_date"].max()
    banned = estimates["trade_date"].eq(asof) & estimates["instrument"].eq("000000.SZ")
    estimates.loc[banned, "a_est"] = np.nan
    coefficient, buy_forbidden, diagnostics = coefficients_at(estimates, asof, 1e7, estimator)
    assert buy_forbidden["000000.SZ"] and buy_forbidden.sum() == 1
    assert diagnostics["amount_unavailable"] == 1
    usable = coefficient.drop("000000.SZ")
    assert coefficient["000000.SZ"] == pytest.approx(usable.quantile(0.90))
    assert coefficient["000000.SZ"] > 0  # never priced at zero


def test_no_usable_cross_section_stops_rather_than_pricing_zero():
    estimator = ImpactEstimator(k=0.5)
    estimates = rolling_estimates(market_panel(), estimator)
    asof = estimates["trade_date"].max()
    estimates.loc[estimates["trade_date"].eq(asof), "a_est"] = np.nan
    with pytest.raises(D12Error, match="usable turnover"):
        coefficients_at(estimates, asof, 1e7, estimator)


# --- nothing after T reaches the estimate at T --------------------------------------

def test_estimates_at_t_ignore_every_later_observation():
    """§6.3's perturbation check: rewrite the future, the T cross-section must not move."""
    estimator = ImpactEstimator.from_config(CONFIG)
    market = market_panel()
    asof = pd.Timestamp(sorted(market["trade_date"].unique())[40])
    future = market.copy()
    later = future["trade_date"] > asof
    future.loc[later, "close"] *= 3.0
    future.loc[later, "daily_amount_cny"] = 1.0
    baseline = rolling_estimates(market, estimator)
    perturbed = rolling_estimates(future, estimator)
    for frame in (baseline, perturbed):
        frame.query("trade_date <= @asof", inplace=True)
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), perturbed.reset_index(drop=True))


def test_sigma_at_t_is_the_volatility_execution_charges_at_t_plus_one():
    """The estimate reuses the execution statistic without lagging it a second time.

    add_ex_ante_volatility shifts its rolling standard deviation one trade day, so the
    value charged on T+1 is the window ending at T; rolling_estimates computes that window
    at T and does not shift. On a balanced panel the two agree exactly. They can differ in
    the median-filled tier when T+1 lists names absent at T, which is the cost of using
    only what is observable at T.
    """
    from scripts.d3_pipeline import add_ex_ante_volatility

    estimator = ImpactEstimator.from_config(CONFIG)
    market = market_panel(days=80, names=8, seed=5)
    charged = add_ex_ante_volatility(market.assign(vwap=market["close"]), CONFIG)
    estimated = rolling_estimates(market, estimator)
    dates = list(pd.DatetimeIndex(sorted(market["trade_date"].unique())))
    merged = (estimated.assign(execution_date=lambda f: f["trade_date"].map(
        dict(zip(dates[:-1], dates[1:]))))
        .merge(charged[["trade_date", "instrument", "volatility"]],
               left_on=["execution_date", "instrument"], right_on=["trade_date", "instrument"],
               suffixes=("", "_exec")).dropna(subset=["execution_date"]))
    assert len(merged) > 400
    np.testing.assert_allclose(merged["sigma_est"], merged["volatility"], rtol=0, atol=0)


def test_short_history_falls_back_and_says_which_tier_it_used():
    """18 sessions clear the 15-observation turnover rule but not the 20-day sigma rule."""
    estimator = ImpactEstimator.from_config(CONFIG)
    estimates = rolling_estimates(market_panel(days=18, names=4), estimator)
    assert set(estimates["sigma_source"]) == {"constant_fallback"}
    assert (estimates["sigma_est"] == CONFIG["costs"]["impact"]["fallback_daily_volatility"]).all()
    assert estimates["a_est"].notna().any()
    assert estimates.loc[estimates["a_observations"] < 15, "a_est"].isna().all()
    assert estimates.loc[estimates["a_observations"] >= 15, "a_est"].notna().all()


def test_the_estimate_does_not_know_about_the_participation_cap():
    """A documented gap, not a defect: the two sides guard different things.

    trade_cost refuses a fill above 5% of the day's turnover, and execute_orders caps the
    quantity before it ever gets there. The decision-side term has no such cap, so a large
    proposed trade is priced as if it would fill whole. Whether R1 needs its own cap is a
    Day-4 question; today it only has to be recorded.
    """
    model, nav, amount = cost_model(), 1e7, 1e8
    over_cap = 0.06 * amount / nav
    assert impact_fraction_of_trade(over_cap, 0.02, amount, nav, 0.5) > 0
    with pytest.raises(ValueError, match="上限"):
        trade_cost(over_cap * nav, 0.02, amount, EXECUTION_DATE, model)


def test_turnover_estimate_needs_fifteen_observations():
    estimator = ImpactEstimator.from_config(CONFIG)
    market = market_panel(days=30, names=3)
    holes = market.groupby("instrument").head(20).index[::2]
    market.loc[holes, "daily_amount_cny"] = np.nan
    estimates = rolling_estimates(market, estimator)
    joined = estimates.set_index(["trade_date", "instrument"])
    assert joined.loc[joined["a_observations"] < 15, "a_est"].isna().all()
    assert joined.loc[joined["a_observations"] >= 15, "a_est"].notna().all()


# --- the penalty is convex, and CLARABEL will take it --------------------------------

def optimizer_fixture(names=40, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"{i:06d}.SZ" for i in range(names)], name="instrument")
    industries = [f"IND_{i}" for i in range(4)]
    exposures = pd.DataFrame(0.0, index=index, columns=[*industries, "Size", "Beta"])
    for position, name in enumerate(index):
        exposures.loc[name, industries[position % len(industries)]] = 1.0
    exposures["Size"] = rng.normal(size=names)
    exposures["Beta"] = rng.normal(size=names)
    benchmark = pd.Series(1 / names, index=index)
    return {"mu": pd.Series(rng.normal(0, 0.01, size=names), index=index),
            "benchmark": benchmark, "pretrade": benchmark.copy(), "exposures": exposures,
            "buy_allowed": pd.Series(True, index=index), "industries": industries,
            "styles": ["Size", "Beta"],
            "limits": OptimizationLimits(stock_active=0.015, industry_active=0.02,
                                         style_active=0.3, single_sided_turnover=0.15),
            "costs": cost_model(), "execution_date": EXECUTION_DATE}


@pytest.mark.parametrize("mode", ["abs", "split"])
def test_penalty_is_dcp_and_evaluates_to_the_registered_formula(mode):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(11)
    old = rng.uniform(0, 0.05, size=25)
    coefficient = rng.uniform(0.001, 0.05, size=25)
    variable = cp.Variable(25)
    penalty = impact_penalty(variable, old, coefficient, mode=mode)
    assert penalty.is_dcp() and penalty.is_convex()
    point = old + rng.uniform(-0.01, 0.01, size=25)
    variable.value = point
    expected = expected_impact_fraction(pd.Series(point - old), pd.Series(coefficient))
    assert float(penalty.value) == pytest.approx(expected, rel=1e-10)
    # The default is the spelling CLARABEL solves cleanly; see d12_solver_probe.py.
    default = impact_penalty(variable, old, coefficient)
    assert float(default.value) == pytest.approx(expected, rel=1e-10)


def test_clarabel_solves_the_penalised_problem_and_it_expects_less_impact():
    """Feasibility probe, not an R1 result: same constraints, penalty on or off."""
    cp = pytest.importorskip("cvxpy")
    setup = optimizer_fixture()
    index = setup["benchmark"].index
    old = setup["pretrade"].to_numpy()
    estimator = ImpactEstimator(k=0.5)
    rng = np.random.default_rng(7)
    coefficient = pd.Series(0.5 * rng.uniform(0.01, 0.04, size=len(index))
                            * np.sqrt(1e7 / rng.uniform(5e7, 5e8, size=len(index))), index=index)

    def solve(penalised):
        w = cp.Variable(len(index))
        active = w - setup["benchmark"].to_numpy()
        objective = setup["mu"].to_numpy() @ active
        if penalised:
            objective = objective - impact_penalty(w, old, coefficient.to_numpy())
        constraints = [cp.sum(w) == 1, w >= 0,
                       active <= setup["limits"].stock_active,
                       active >= -setup["limits"].stock_active,
                       0.5 * cp.norm1(w - old) <= setup["limits"].single_sided_turnover]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver="CLARABEL")
        assert problem.status == "optimal"
        return pd.Series(np.maximum(w.value, 0.0), index=index)

    plain, penalised = solve(False), solve(True)
    for target in (plain, penalised):
        assert 0.5 * float((target - setup["pretrade"]).abs().sum()) <= 0.15 + 1e-6
    assert (expected_impact_fraction(penalised - setup["pretrade"], coefficient)
            < expected_impact_fraction(plain - setup["pretrade"], coefficient))
    assert estimator.k == 0.5  # the probe used the registered coefficient, not a fitted one


# --- the D12 guard mirrors the D11 one ----------------------------------------------

def test_required_inputs_mirror_does_not_drift_from_d11():
    from src.d11_io import required_inputs as d11_required_inputs

    assert required_inputs(CONFIG) == d11_required_inputs(CONFIG)


def test_verify_repository_refuses_a_branch_outside_the_allow_list():
    baseline = yaml.safe_load((ROOT / "config/d11_baseline.json").read_text())
    with pytest.raises(D12Error, match="requires one of"):
        verify_repository(ROOT, baseline, branches=("no-such-branch",))


def test_impact_estimator_reads_the_registered_execution_parameters():
    estimator = ImpactEstimator.from_config(CONFIG)
    impact = CONFIG["costs"]["impact"]
    assert estimator.k == impact["coefficient"] == 0.5
    assert estimator.volatility_lookback == impact["volatility_lookback_trade_days"]
    assert estimator.volatility_minimum_observations == impact["minimum_observations"]
    assert estimator.fallback_daily_volatility == impact["fallback_daily_volatility"]
    assert estimator.amount_lookback == CONFIG["tradability"]["amount_lookback_trade_days"]
    assert ImpactEstimator.from_config(CONFIG, k=0.0).bypassed
    with pytest.raises(D12Error):
        ImpactEstimator(k=-1.0).validate()


def test_estimator_rejects_a_market_frame_it_cannot_trust():
    estimator = ImpactEstimator.from_config(CONFIG)
    with pytest.raises(D12Error, match="missing"):
        rolling_estimates(market_panel().drop(columns=["daily_amount_cny"]), estimator)
    doubled = pd.concat([market_panel(days=25, names=2)] * 2, ignore_index=True)
    with pytest.raises(D12Error, match="duplicate"):
        rolling_estimates(doubled, estimator)
    estimates = rolling_estimates(market_panel(), estimator)
    with pytest.raises(D12Error, match="NAV"):
        coefficients_at(estimates, estimates["trade_date"].max(), 0.0, estimator)

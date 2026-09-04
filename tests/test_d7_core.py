import numpy as np
import pandas as pd
import pytest

from scripts.d7_pipeline import (
    build_execution_price_matrix,
    holding_returns_between,
    make_execution_schedule,
)
from src.d7_core import (
    OptimizationLimits,
    PortfolioOptimizationError,
    annualized_tracking_error,
    audit_portfolio_constraints,
    drift_pretrade_weights,
    optimize_long_only_active_portfolio,
    symmetric_l1_cost_rate,
)


def synthetic_problem() -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    instruments = pd.Index([f"S{i:02d}" for i in range(12)], name="instrument")
    benchmark = pd.Series(1.0 / len(instruments), index=instruments)
    pretrade = benchmark.copy()
    mu = pd.Series(np.linspace(-0.004, 0.004, len(instruments)), index=instruments)
    exposures = pd.DataFrame(index=instruments)
    for number in range(3):
        exposures[f"industry_{number}"] = (
            np.arange(len(instruments)) // 4 == number
        ).astype(float)
    size = np.linspace(-1.5, 1.5, len(instruments))
    exposures["style_Size"] = (size - size.mean()) / size.std(ddof=0)
    beta = np.tile([-1.0, 1.0], len(instruments) // 2)
    exposures["style_Beta"] = (beta - beta.mean()) / beta.std(ddof=0)
    return mu, benchmark, pretrade, exposures


def plan_limits() -> OptimizationLimits:
    return OptimizationLimits(
        stock_active=0.015,
        industry_active=0.02,
        style_active=0.30,
        single_sided_turnover=0.15,
        tolerance=1e-6,
    )


def test_required_solver_is_available() -> None:
    try:
        import cvxpy as cp
    except ImportError:
        pytest.fail("D7 需要先安装 cvxpy 与 clarabel")
    assert "CLARABEL" in cp.installed_solvers()


def test_symmetric_l1_cost_is_exact_for_fully_invested_rebalance() -> None:
    rate = symmetric_l1_cost_rate(0.00015, 0.00001, 0.001)
    assert rate == pytest.approx(0.00066)
    buys = 0.12
    sells = 0.12
    direct_cost = buys * 0.00016 + sells * 0.00116
    assert rate * (buys + sells) == pytest.approx(direct_cost)


def test_optimizer_satisfies_every_planned_hard_constraint() -> None:
    mu, benchmark, pretrade, exposures = synthetic_problem()
    result = optimize_long_only_active_portfolio(
        mu,
        benchmark,
        pretrade,
        exposures,
        ["industry_0", "industry_1", "industry_2"],
        ["style_Size", "style_Beta"],
        cost_rate=0.00066,
        limits=plan_limits(),
    )
    assert result.status == "optimal"
    assert result.audit["passed"]
    assert result.audit["maximum_stock_active"] <= 0.015 + 1e-6
    assert result.audit["maximum_industry_active"] <= 0.02 + 1e-6
    assert result.audit["maximum_style_active"] <= 0.30 + 1e-6
    assert result.audit["single_sided_turnover"] <= 0.15 + 1e-6
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert result.weights.min() >= -1e-8


def test_cost_in_objective_can_prevent_low_conviction_turnover() -> None:
    mu, benchmark, pretrade, exposures = synthetic_problem()
    low_conviction = mu * 0.01
    no_cost = optimize_long_only_active_portfolio(
        low_conviction,
        benchmark,
        pretrade,
        exposures,
        ["industry_0", "industry_1", "industry_2"],
        ["style_Size", "style_Beta"],
        cost_rate=0.0,
        limits=plan_limits(),
    )
    with_cost = optimize_long_only_active_portfolio(
        low_conviction,
        benchmark,
        pretrade,
        exposures,
        ["industry_0", "industry_1", "industry_2"],
        ["style_Size", "style_Beta"],
        cost_rate=0.00066,
        limits=plan_limits(),
    )
    assert no_cost.audit["single_sided_turnover"] > 0.01
    assert with_cost.audit["single_sided_turnover"] < 1e-6


def test_optimizer_has_no_infeasible_fallback() -> None:
    mu, benchmark, _, exposures = synthetic_problem()
    pretrade = pd.Series({"REMOVED": 1.0})
    with pytest.raises(PortfolioOptimizationError, match="禁止 fallback"):
        optimize_long_only_active_portfolio(
            mu,
            benchmark,
            pretrade,
            exposures,
            ["industry_0", "industry_1", "industry_2"],
            ["style_Size", "style_Beta"],
            cost_rate=0.00066,
            limits=plan_limits(),
        )


def test_pretrade_weights_drift_and_renormalize() -> None:
    target = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.10, "B": -0.05})
    result = drift_pretrade_weights(target, returns)
    expected_a = 0.66 / (0.66 + 0.38)
    assert result["A"] == pytest.approx(expected_a)
    assert result.sum() == pytest.approx(1.0)


def test_pretrade_drift_fails_on_missing_held_return() -> None:
    with pytest.raises(PortfolioOptimizationError, match="缺少收益"):
        drift_pretrade_weights(
            pd.Series({"A": 0.6, "B": 0.4}), pd.Series({"A": 0.01})
        )


def test_constraint_audit_detects_actual_stock_violation() -> None:
    _, benchmark, pretrade, exposures = synthetic_problem()
    bad = benchmark.copy()
    bad.iloc[0] += 0.03
    bad.iloc[1] -= 0.03
    audit = audit_portfolio_constraints(
        bad,
        benchmark,
        pretrade,
        exposures,
        ["industry_0", "industry_1", "industry_2"],
        ["style_Size", "style_Beta"],
        plan_limits(),
    )
    assert not audit["checks"]["stock_active"]
    assert not audit["passed"]


def test_tracking_error_reports_partial_covariance_coverage() -> None:
    active = pd.Series({"A": 0.02, "B": -0.01, "C": -0.01})
    covariance = np.diag([0.0001, 0.0004])
    result = annualized_tracking_error(
        active,
        covariance,
        ["A", "B"],
        annualization_factor=252,
    )
    expected_variance = 0.02**2 * 0.0001 + (-0.01) ** 2 * 0.0004
    assert result.annualized_tracking_error == pytest.approx(
        np.sqrt(expected_variance * 252)
    )
    assert result.active_l1_coverage == pytest.approx(0.75)
    assert result.excluded_instruments == ("C",)


def test_execution_schedule_uses_the_next_actual_trading_day() -> None:
    signals = pd.DatetimeIndex(["2018-09-28", "2018-10-12"])
    calendar = pd.DatetimeIndex(
        ["2018-09-28", "2018-10-08", "2018-10-12", "2018-10-15"]
    )
    result = make_execution_schedule(signals, calendar)
    assert result["execution_date"].tolist() == [
        pd.Timestamp("2018-10-08"),
        pd.Timestamp("2018-10-15"),
    ]


def test_holding_returns_use_execution_to_execution_prices() -> None:
    prices = pd.DataFrame(
        {"A": [10.0, 11.0], "B": [20.0, 19.0]},
        index=pd.to_datetime(["2018-01-08", "2018-01-15"]),
    )
    result = holding_returns_between(
        prices,
        "2018-01-08",
        "2018-01-15",
        pd.Index(["A", "B"]),
    )
    assert result["A"] == pytest.approx(0.10)
    assert result["B"] == pytest.approx(-0.05)


def test_suspended_day_uses_latest_past_close_without_future_leakage() -> None:
    dates = pd.to_datetime(["2018-01-08", "2018-01-09", "2018-01-10"])
    prices = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 2),
            "instrument": ["A", "B"] * 3,
            "vwap": [9.8, 20.0, np.nan, 20.2, 12.1, 20.4],
            "close": [10.0, 20.0, np.nan, 20.2, 12.0, 20.4],
        }
    )
    matrix = build_execution_price_matrix(prices)
    assert matrix.loc[pd.Timestamp("2018-01-09"), "A"] == pytest.approx(10.0)
    assert matrix.loc[pd.Timestamp("2018-01-09"), "A"] != pytest.approx(12.0)
    holding_return = holding_returns_between(
        matrix,
        "2018-01-08",
        "2018-01-09",
        pd.Index(["A"]),
    )
    assert holding_return["A"] == pytest.approx(10.0 / 9.8 - 1.0)

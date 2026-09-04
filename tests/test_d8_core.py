import numpy as np
import pandas as pd
import pytest

from scripts.d8_pipeline import (
    build_frozen_d7_pretrade_path,
    optimizer_limits_for_layer,
    run_loss_ladder_backtests,
)
from src.d8_core import (
    LossLadderError,
    assert_layer6_matches_d7,
    audit_target_exposures,
    build_loss_deltas,
    make_ideal_execution_market,
    make_top_n_equal_targets,
    require_d7_structural_completion,
    summarize_backtest_performance,
    validate_layer_ids,
)


def test_layer_ids_are_fixed_to_seven_ordered_layers() -> None:
    assert validate_layer_ids(range(7)) == tuple(range(7))
    with pytest.raises(ValueError, match="0–6"):
        validate_layer_ids([0, 1, 3, 2, 4, 5, 6])


def test_top_n_equal_targets_are_deterministic_on_ties() -> None:
    predictions = pd.DataFrame(
        {
            "sample_date": [pd.Timestamp("2018-01-05")] * 5,
            "instrument": ["E", "D", "C", "B", "A"],
            "mu": [0.01, 0.02, 0.02, -0.01, 0.03],
        }
    )
    result = make_top_n_equal_targets(predictions, top_n=3, score_col="mu")
    assert result["instrument"].tolist() == ["A", "C", "D"]
    assert result["target_weight"].tolist() == pytest.approx([1 / 3] * 3)
    assert result["target_weight"].sum() == pytest.approx(1.0)


def test_target_exposure_audit_measures_actual_active_constraints() -> None:
    instruments = pd.Index(["A", "B", "C", "D"])
    benchmark = pd.Series(0.25, index=instruments)
    target = pd.Series([0.30, 0.20, 0.25, 0.25], index=instruments)
    exposures = pd.DataFrame(
        {
            "industry_1": [1.0, 1.0, 0.0, 0.0],
            "industry_2": [0.0, 0.0, 1.0, 1.0],
            "style_Size": [-1.0, 1.0, -1.0, 1.0],
            "style_Beta": [1.0, 1.0, -1.0, -1.0],
        },
        index=instruments,
    )
    audit = audit_target_exposures(
        target,
        benchmark,
        exposures,
        ["industry_1", "industry_2"],
        ["style_Size", "style_Beta"],
    )
    assert audit.maximum_stock_active == pytest.approx(0.05)
    assert audit.maximum_industry_active == pytest.approx(0.0)
    assert audit.style_active_exposures["style_Size"] == pytest.approx(-0.10)


def test_performance_summary_keeps_initial_transaction_cost() -> None:
    dates = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04"])
    nav = pd.DataFrame(
        {
            "trade_date": dates,
            "nav": [1000.0, 990.0, 1089.0],
            "gross_exposure": [0.0, 1.0, 1.0],
            "single_sided_turnover": [0.0, 0.5, 0.0],
            "transaction_cost": [0.0, 10.0, 0.0],
            "signal_date": [pd.NaT, dates[0], pd.NaT],
        }
    )
    benchmark = pd.DataFrame(
        {"trade_date": dates, "close": [100.0, 100.0, 110.0]}
    )
    metrics, daily = summarize_backtest_performance(
        nav,
        benchmark,
        initial_capital=1000.0,
        annualization_factor=1.0,
    )
    assert daily["portfolio_return"].iloc[0] == pytest.approx(-0.01)
    assert metrics["ending_nav"] == pytest.approx(1089.0)
    assert metrics["portfolio_annualized_return"] == pytest.approx(0.089)
    assert metrics["benchmark_annualized_return"] == pytest.approx(0.10)
    assert metrics["annualized_excess_return"] == pytest.approx(-0.011)
    assert metrics["total_transaction_cost"] == pytest.approx(10.0)


def test_layer6_must_match_frozen_d7_weights_exactly() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2018-01-05", "2018-01-05"]),
            "instrument": ["A", "B"],
            "target_weight": [0.4, 0.6],
        }
    )
    audit = assert_layer6_matches_d7(frame.copy(), frame.copy(), tolerance=1e-12)
    assert audit["passed"]
    changed = frame.copy()
    changed.loc[0, "target_weight"] += 1e-4
    with pytest.raises(LossLadderError, match="未精确复用"):
        assert_layer6_matches_d7(changed, frame, tolerance=1e-12)


def d7_summary(*, solver_passed: bool = True) -> dict:
    return {
        "mode": "design",
        "oos_data_accessed": False,
        "acceptance": {
            "upstream_d6": True,
            "all_available_design_signal_dates": True,
            "solver_success_100pct": solver_passed,
            "fallback_zero": True,
            "fully_invested": True,
            "long_only": True,
            "stock_active": True,
            "industry_active": True,
            "style_active": True,
            "turnover": True,
            "covariance_coverage": True,
            "tracking_error": False,
            "passed": False,
        },
    }


def test_known_d7_te_miss_does_not_block_d8_but_structural_failure_does() -> None:
    status = require_d7_structural_completion(d7_summary())
    assert status["structural_checks_passed"]
    assert not status["tracking_error_passed"]
    with pytest.raises(LossLadderError, match="solver_success_100pct"):
        require_d7_structural_completion(d7_summary(solver_passed=False))
    unexpected = d7_summary()
    unexpected["acceptance"]["new_structural_check"] = False
    with pytest.raises(LossLadderError, match="new_structural_check"):
        require_d7_structural_completion(unexpected)


def test_loss_deltas_allow_neutralization_to_improve_return() -> None:
    ladder = pd.DataFrame(
        {
            "capital": [10.0] * 7,
            "layer": list(range(7)),
            "annualized_excess_return": [0.10, 0.08, 0.07, 0.09, 0.085, 0.06, 0.055],
        }
    )
    result = build_loss_deltas(ladder, capital=10.0)
    assert result.loc[result["layer"].eq(3), "change_vs_previous"].iloc[0] == pytest.approx(
        0.02
    )
    assert result.loc[result["layer"].ge(1), "explanation"].ne("").all()


def test_ideal_market_fills_amount_causally_and_enables_both_sides() -> None:
    dates = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04"])
    market = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 2),
            "instrument": ["A", "B"] * 3,
            "close": [10.0, 20.0, np.nan, 20.0, 12.0, 20.0],
            "vwap": [10.0, 20.0, np.nan, 20.0, 12.0, 20.0],
            "daily_amount_cny": [100.0, 200.0, np.nan, 210.0, 1000.0, 220.0],
            "volatility": [0.02] * 6,
            "can_buy": [True, True, False, True, True, True],
            "can_sell": [True, True, False, True, True, True],
        }
    )
    result = make_ideal_execution_market(market, fallback_daily_amount=50.0)
    suspended = result.loc[
        result["trade_date"].eq(pd.Timestamp("2018-01-03"))
        & result["instrument"].eq("A")
    ].iloc[0]
    assert suspended["daily_amount_cny"] == pytest.approx(100.0)
    assert suspended["daily_amount_cny"] != pytest.approx(1000.0)
    assert bool(suspended["can_buy"])
    assert bool(suspended["can_sell"])


def test_optimizer_profiles_add_constraints_in_the_planned_order() -> None:
    config = {
        "d7": {
            "constraint_tolerance": 1e-6,
            "stock_active_limit": 0.015,
            "industry_active_limit": 0.02,
            "style_active_limit": 0.30,
            "single_sided_turnover_limit": 0.15,
            "constrained_style_factors": ["Size", "Beta"],
        }
    }
    layer3, styles3 = optimizer_limits_for_layer(config, 3)
    layer4, styles4 = optimizer_limits_for_layer(config, 4)
    layer5, styles5 = optimizer_limits_for_layer(config, 5)
    assert layer3.industry_active == pytest.approx(0.02)
    assert layer3.style_active == pytest.approx(1.0)
    assert styles3 == []
    assert layer4.style_active == pytest.approx(0.30)
    assert styles4 == ["style_Size", "style_Beta"]
    assert layer5.single_sided_turnover == pytest.approx(0.15)
    assert layer5.stock_active == pytest.approx(1.0)
    assert styles5 == styles4


def test_frozen_d7_pretrade_path_is_reconstructed_causally() -> None:
    signal_dates = pd.DatetimeIndex(["2018-01-02", "2018-01-04"])
    execution_by_signal = pd.Series(
        pd.to_datetime(["2018-01-03", "2018-01-05"]),
        index=signal_dates,
    )
    d7_targets = pd.DataFrame(
        {
            "signal_date": np.repeat(signal_dates, 2),
            "instrument": ["A", "B", "A", "B"],
            "target_weight": [0.6, 0.4, 0.5, 0.5],
        }
    )
    exposures = pd.DataFrame(
        {
            "trade_date": np.repeat(signal_dates, 2),
            "instrument": ["A", "B", "A", "B"],
            "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    price_matrix = pd.DataFrame(
        {"A": [10.0, 11.0], "B": [20.0, 18.0]},
        index=pd.to_datetime(["2018-01-03", "2018-01-05"]),
    )
    result = build_frozen_d7_pretrade_path(
        d7_targets,
        exposures,
        signal_dates,
        execution_by_signal,
        price_matrix,
        tolerance=1e-6,
    )
    assert result[signal_dates[0]].tolist() == pytest.approx([0.5, 0.5])
    expected_a = 0.6 * 1.1 / (0.6 * 1.1 + 0.4 * 0.9)
    assert result[signal_dates[1]].loc["A"] == pytest.approx(expected_a)


def test_backtest_layers_isolate_cost_then_enable_tradability() -> None:
    dates = pd.date_range("2018-01-02", periods=4, freq="D")
    rows = []
    for date in dates:
        for instrument, price in [("A", 10.0), ("B", 20.0)]:
            rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "close": price,
                    "vwap": price,
                    "daily_amount_cny": 1_000_000.0,
                    "volatility": 0.02,
                    "can_buy": True,
                    "can_sell": True,
                }
            )
    ideal = pd.DataFrame(rows)
    strategy = ideal.copy()
    strategy.loc[
        strategy["trade_date"].eq(dates[1])
        & strategy["instrument"].eq("A"),
        "can_buy",
    ] = False
    targets = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[0]],
            "instrument": ["A", "B"],
            "target_weight": [0.5, 0.5],
        }
    )
    config = {
        "portfolio": {"capital_scenarios": [10_000.0]},
        "execution": {"participation_limit": 0.05},
        "costs": {
            "commission_rate": 0.00015,
            "transfer_fee_rate": 0.00001,
            "stamp_duty": [
                {
                    "start_date": "2014-01-01",
                    "end_date": "2023-08-27",
                    "rate": 0.001,
                }
            ],
            "impact": {"coefficient": 0.5},
        },
        "d8": {
            "ideal_participation_limit": 1.0,
            "return_annualization_factor": 252,
        },
    }
    ladder, _, _ = run_loss_ladder_backtests(
        {layer: targets.copy() for layer in range(7)},
        ideal,
        ideal,
        strategy,
        pd.DataFrame({"trade_date": dates, "close": [100.0] * len(dates)}),
        pd.DatetimeIndex(dates),
        config,
    )
    result = ladder.set_index("layer")
    assert result.loc[0, "total_transaction_cost"] == pytest.approx(0.0)
    assert result.loc[1, "total_transaction_cost"] > 0
    assert result.loc[1, "impact_cost"] > 0
    assert result.loc[0, "blocked_by_tradability_rows"] == 0
    assert result.loc[1, "blocked_by_tradability_rows"] == 0
    assert result.loc[2, "blocked_by_tradability_rows"] > 0

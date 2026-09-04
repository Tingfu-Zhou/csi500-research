from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from src.d3_core import (
    CostModel,
    LeakageViolation,
    assert_feature_timestamps,
    assert_future_price_invariance,
    build_causal_price_features,
    make_label_windows,
    match_target_orders,
    run_vwap_backtest,
    trade_cost,
)


def cost_model() -> CostModel:
    return CostModel(
        commission_rate=0.00015,
        transfer_fee_rate=0.00001,
        stamp_duty=(
            (pd.Timestamp("2014-01-01"), pd.Timestamp("2023-08-27"), 0.001),
            (pd.Timestamp("2023-08-28"), pd.Timestamp("2026-08-31"), 0.0005),
        ),
        impact_coefficient=0.5,
        participation_limit=0.05,
    )


def zero_cost_model() -> CostModel:
    return CostModel.zero_cost(participation_limit=0.05)


def quote_frame(
    instruments: list[str],
    *,
    daily_amount: float = 10_000_000.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "execution_price": 10.0,
            "daily_amount_cny": daily_amount,
            "volatility": volatility,
            "can_buy": True,
            "can_sell": True,
        },
        index=instruments,
    )


def test_cost_math_and_sell_only_stamp_duty() -> None:
    model = cost_model()
    buy = trade_cost(100_000, 0.02, 10_000_000, "2023-08-27", model)
    old_sell = trade_cost(-100_000, 0.02, 10_000_000, "2023-08-27", model)
    new_sell = trade_cost(-100_000, 0.02, 10_000_000, "2023-08-28", model)

    assert buy["commission_cost"] == pytest.approx(15.0)
    assert buy["transfer_fee_cost"] == pytest.approx(1.0)
    assert buy["impact_rate"] == pytest.approx(0.001)
    assert buy["impact_cost"] == pytest.approx(100.0)
    assert buy["stamp_duty_cost"] == 0.0
    assert buy["total_cost"] == pytest.approx(116.0)
    assert old_sell["stamp_duty_cost"] == pytest.approx(100.0)
    assert old_sell["total_cost"] == pytest.approx(216.0)
    assert new_sell["stamp_duty_cost"] == pytest.approx(50.0)
    assert new_sell["total_cost"] == pytest.approx(166.0)


def test_stamp_duty_schedule_must_cover_trade_date() -> None:
    with pytest.raises(ValueError, match="未命中唯一印花税区间"):
        cost_model().stamp_rate("2013-12-31")


def test_participation_cap_limits_trade_amount() -> None:
    market = quote_frame(["A"], daily_amount=1_000_000.0)
    result = match_target_orders(
        pd.Series({"A": 0.0}),
        cash=100_000.0,
        target_weights=pd.Series({"A": 1.0}),
        market=market,
        trade_date="2018-01-02",
        model=zero_cost_model(),
    )
    row = result.trades.set_index("instrument").loc["A"]
    assert row["executed_trade_amount"] == pytest.approx(50_000.0)
    assert bool(row["participation_limited"])
    assert result.cash_after == pytest.approx(50_000.0)


def test_float32_daily_amount_does_not_create_false_capacity_breach() -> None:
    daily_amount = np.float32(57_932_144.0)
    market = quote_frame(["A"], daily_amount=float(daily_amount))
    market["daily_amount_cny"] = market["daily_amount_cny"].astype("float32")
    result = match_target_orders(
        pd.Series({"A": 0.0}),
        cash=20_000_000.0,
        target_weights=pd.Series({"A": 1.0}),
        market=market,
        trade_date="2018-01-02",
        model=cost_model(),
    )
    executed = float(result.trades.set_index("instrument").loc["A", "executed_trade_amount"])
    exact_capacity = float(daily_amount) * cost_model().participation_limit
    assert executed <= exact_capacity + 1e-8
    assert executed == pytest.approx(exact_capacity)


@pytest.mark.parametrize(
    ("side", "can_buy", "can_sell"),
    [("buy", False, True), ("sell", True, False)],
)
def test_suspension_or_price_limit_blocks_correct_side(
    side: str, can_buy: bool, can_sell: bool
) -> None:
    market = quote_frame(["A"])
    market.loc["A", "can_buy"] = can_buy
    market.loc["A", "can_sell"] = can_sell
    if side == "buy":
        positions = pd.Series({"A": 0.0})
        cash = 100_000.0
        target = pd.Series({"A": 1.0})
    else:
        positions = pd.Series({"A": 100_000.0})
        cash = 0.0
        target = pd.Series({"A": 0.0})
    result = match_target_orders(
        positions, cash, target, market, "2018-01-02", zero_cost_model()
    )
    row = result.trades.set_index("instrument").loc["A"]
    assert row["executed_trade_amount"] == 0.0
    assert bool(row["blocked_by_tradability"])


def test_two_stock_vwap_backtest_matches_hand_calculation() -> None:
    dates = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04"])
    rows = []
    for date, a_price, b_price in zip(dates, [10.0, 10.0, 11.0], [20.0, 20.0, 18.0]):
        rows.extend(
            [
                {
                    "trade_date": date,
                    "instrument": "A",
                    "close": a_price,
                    "vwap": a_price,
                    "daily_amount_cny": 1_000_000,
                    "volatility": 0.02,
                    "can_buy": True,
                    "can_sell": True,
                },
                {
                    "trade_date": date,
                    "instrument": "B",
                    "close": b_price,
                    "vwap": b_price,
                    "daily_amount_cny": 1_000_000,
                    "volatility": 0.02,
                    "can_buy": True,
                    "can_sell": True,
                },
            ]
        )
    target = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[0]],
            "instrument": ["A", "B"],
            "target_weight": [0.5, 0.5],
        }
    )
    result = run_vwap_backtest(
        pd.DataFrame(rows), target, initial_capital=100_000, model=zero_cost_model()
    )
    nav = result.nav.set_index("trade_date")["nav"]
    assert nav.loc[dates[0]] == pytest.approx(100_000.0)
    assert nav.loc[dates[1]] == pytest.approx(100_000.0)
    assert nav.loc[dates[2]] == pytest.approx(100_000.0)
    assert result.trades["execution_date"].nunique() == 1
    assert result.trades["execution_date"].iloc[0] == dates[1]


def test_initial_buy_cost_is_deducted_from_nav_and_cash_never_goes_negative() -> None:
    dates = pd.to_datetime(["2018-01-02", "2018-01-03"])
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "instrument": ["A", "A"],
            "close": [10.0, 10.0],
            "vwap": [10.0, 10.0],
            "daily_amount_cny": [10_000_000.0, 10_000_000.0],
            "volatility": [0.0, 0.0],
            "can_buy": [True, True],
            "can_sell": [True, True],
        }
    )
    target = pd.DataFrame(
        {
            "signal_date": [dates[0]],
            "instrument": ["A"],
            "target_weight": [1.0],
        }
    )
    model = CostModel(
        commission_rate=0.00015,
        transfer_fee_rate=0.00001,
        stamp_duty=((pd.Timestamp("2014-01-01"), pd.Timestamp("2018-12-31"), 0.001),),
        impact_coefficient=0.0,
        participation_limit=0.05,
    )
    result = run_vwap_backtest(market, target, 100_000.0, model)
    expected_notional = 100_000.0 / 1.00016
    expected_cost = expected_notional * 0.00016
    final = result.nav.set_index("trade_date").loc[dates[1]]
    assert final["cash"] == pytest.approx(0.0, abs=1e-8)
    assert final["transaction_cost"] == pytest.approx(expected_cost)
    assert final["nav"] == pytest.approx(100_000.0 - expected_cost)


def test_large_capital_sub_cent_negative_cash_is_clamped_but_debt_is_rejected() -> None:
    market = quote_frame(["A"], daily_amount=100_000_000.0)
    positions = pd.Series({"A": 20_000_000.0})
    target = pd.Series({"A": 1.0})
    residue = match_target_orders(
        positions,
        cash=-3e-8,
        target_weights=target,
        market=market,
        trade_date="2018-01-02",
        model=cost_model(),
    )
    assert residue.cash_after == 0.0
    with pytest.raises(ValueError, match="现金必须是非负有限数"):
        match_target_orders(
            positions,
            cash=-0.01,
            target_weights=target,
            market=market,
            trade_date="2018-01-02",
            model=cost_model(),
        )


def test_suspended_position_is_frozen_and_not_funded_into_replacement() -> None:
    dates = pd.to_datetime(
        ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
    )
    rows = []
    for date in dates:
        for instrument in ["A", "B"]:
            suspended_a = date == dates[3] and instrument == "A"
            rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "close": np.nan if suspended_a else 10.0,
                    "vwap": np.nan if suspended_a else 10.0,
                    "daily_amount_cny": 0.0 if suspended_a else 10_000_000.0,
                    "volatility": 0.02,
                    "can_buy": not suspended_a,
                    "can_sell": not suspended_a,
                }
            )
    targets = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[2]],
            "instrument": ["A", "B"],
            "target_weight": [1.0, 1.0],
        }
    )
    result = run_vwap_backtest(
        pd.DataFrame(rows), targets, 100_000.0, zero_cost_model()
    )
    second = result.trades.loc[result.trades["execution_date"].eq(dates[3])]
    a = second.set_index("instrument").loc["A"]
    b = second.set_index("instrument").loc["B"]
    assert a["executed_trade_amount"] == 0.0
    assert bool(a["blocked_by_tradability"])
    assert b["executed_trade_amount"] == 0.0
    assert bool(b["cash_limited"])
    assert result.nav.set_index("trade_date").loc[dates[3], "nav"] == pytest.approx(
        100_000.0
    )


def feature_source() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=30, freq="D")
    frames = []
    for number, instrument in enumerate(["A", "B"], start=1):
        close = np.linspace(10 * number, 12 * number, len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "instrument": instrument,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "vwap": close * 0.999,
                    "volume": 1_000_000.0,
                    "daily_amount_cny": 10_000_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_feature_timestamp_assertion_passes_and_catches_violation() -> None:
    calendar = pd.date_range("2018-01-01", periods=12, freq="D")
    windows = make_label_windows([calendar[4]], calendar, horizon_trade_days=5)
    metadata = pd.DataFrame(
        {
            "sample_date": [calendar[4]],
            "instrument": ["A"],
            "feature_max_timestamp": [calendar[4]],
        }
    )
    assert_feature_timestamps(metadata, windows)
    metadata.loc[0, "feature_max_timestamp"] = calendar[5]
    with pytest.raises(LeakageViolation, match="特征时间戳越过标签起点"):
        assert_feature_timestamps(metadata, windows)


def test_future_price_perturbation_keeps_causal_features_unchanged() -> None:
    source = feature_source()
    assert_future_price_invariance(
        build_causal_price_features, source, "2018-01-20", seed=7
    )


def test_future_price_perturbation_handles_qlib_float32_without_false_alarm() -> None:
    source = feature_source()
    price_columns = ["open", "high", "low", "close", "vwap"]
    source[price_columns] = source[price_columns].astype("float32")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_future_price_invariance(
            build_causal_price_features, source, "2018-01-20", seed=7
        )
    incompatible_dtype_warnings = [
        item
        for item in caught
        if issubclass(item.category, FutureWarning)
        and "incompatible dtype" in str(item.message)
    ]
    assert incompatible_dtype_warnings == []


def test_future_price_perturbation_catches_centered_window() -> None:
    source = feature_source()

    def leaky_builder(frame: pd.DataFrame) -> pd.DataFrame:
        result = build_causal_price_features(frame)
        ordered = frame.sort_values(["instrument", "trade_date"]).copy()
        ordered["centered_price"] = ordered.groupby("instrument")["close"].transform(
            lambda series: series.rolling(3, center=True, min_periods=1).mean()
        )
        return result.merge(
            ordered[["trade_date", "instrument", "centered_price"]],
            on=["trade_date", "instrument"],
            how="left",
            validate="one_to_one",
        )

    with pytest.raises(LeakageViolation, match="未来价格扰动改变"):
        assert_future_price_invariance(
            leaky_builder, source, "2018-01-20", seed=7
        )

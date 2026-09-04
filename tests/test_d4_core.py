from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.d4_core import (
    build_forward_excess_labels,
    compute_ic_decay,
    cross_sectional_zscore,
    decile_return_table,
    evaluate_acceptance,
    rank_ic_series,
    select_weekly_sample_dates,
    split_design_samples,
    summarize_ic,
    validate_alpha158_spec,
)


def test_weekly_sample_uses_last_trading_day_in_w_fri_week() -> None:
    weights = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2018-01-02", "2018-01-04", "2018-01-08", "2018-01-12"]
            ),
            "instrument": ["A"] * 4,
            "weight": [1.0] * 4,
        }
    )
    result = select_weekly_sample_dates(weights)
    assert result.tolist() == [pd.Timestamp("2018-01-04"), pd.Timestamp("2018-01-12")]


def label_inputs() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    calendar = pd.bdate_range("2018-01-02", periods=10)
    prices = []
    a = np.linspace(10.0, 12.0, len(calendar))
    b = np.linspace(20.0, 18.0, len(calendar))
    for date, a_price, b_price in zip(calendar, a, b):
        prices.extend(
            [
                {"trade_date": date, "instrument": "A", "vwap": a_price},
                {"trade_date": date, "instrument": "B", "vwap": b_price},
            ]
        )
    weights = pd.DataFrame(
        {
            "trade_date": [calendar[0], calendar[0]],
            "instrument": ["A", "B"],
            "weight": [0.6, 0.4],
        }
    )
    return calendar, pd.DataFrame(prices), weights


def test_forward_label_is_t_plus_1_to_t_plus_6_and_weighted_excess() -> None:
    calendar, prices, weights = label_inputs()
    labels, coverage = build_forward_excess_labels(
        prices, weights, [calendar[0]], calendar, [5]
    )
    rows = labels.set_index("instrument")
    a_return = prices.loc[
        prices["instrument"].eq("A") & prices["trade_date"].eq(calendar[6]), "vwap"
    ].iloc[0] / prices.loc[
        prices["instrument"].eq("A") & prices["trade_date"].eq(calendar[1]), "vwap"
    ].iloc[0] - 1
    b_return = prices.loc[
        prices["instrument"].eq("B") & prices["trade_date"].eq(calendar[6]), "vwap"
    ].iloc[0] / prices.loc[
        prices["instrument"].eq("B") & prices["trade_date"].eq(calendar[1]), "vwap"
    ].iloc[0] - 1
    benchmark = 0.6 * a_return + 0.4 * b_return
    assert rows.loc["A", "stock_return"] == pytest.approx(a_return)
    assert rows.loc["B", "stock_return"] == pytest.approx(b_return)
    assert rows.loc["A", "benchmark_return"] == pytest.approx(benchmark)
    assert rows.loc["A", "excess_return"] == pytest.approx(a_return - benchmark)
    assert rows.loc["B", "excess_return"] == pytest.approx(b_return - benchmark)
    assert rows["label_start_timestamp"].unique()[0] == calendar[1]
    assert rows["label_end_timestamp"].unique()[0] == calendar[6]
    assert float(coverage.loc[0, "weight_coverage"]) == pytest.approx(1.0)
    assert float((rows["benchmark_weight"] * rows["excess_return"]).sum()) == pytest.approx(0.0)


def test_missing_endpoint_excludes_only_that_stock_and_audits_coverage() -> None:
    calendar, prices, weights = label_inputs()
    prices.loc[
        prices["instrument"].eq("B") & prices["trade_date"].eq(calendar[6]), "vwap"
    ] = np.nan
    labels, coverage = build_forward_excess_labels(
        prices, weights, [calendar[0]], calendar, [5]
    )
    assert labels["instrument"].tolist() == ["A"]
    assert float(labels.loc[0, "benchmark_return"]) == pytest.approx(
        float(labels.loc[0, "stock_return"])
    )
    assert float(labels.loc[0, "excess_return"]) == pytest.approx(0.0)
    assert float(coverage.loc[0, "weight_coverage"]) == pytest.approx(0.6)
    assert int(coverage.loc[0, "valid_label_count"]) == 1


def test_split_purges_any_training_label_crossing_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(["2016-12-16", "2016-12-23", "2017-01-06"]),
            "label_end_timestamp": pd.to_datetime(["2016-12-23", "2017-01-03", "2017-01-13"]),
            "instrument": ["A", "A", "A"],
        }
    )
    train, validation = split_design_samples(
        frame,
        train_start="2014-01-01",
        train_end="2016-12-31",
        validation_start="2017-01-01",
        validation_end="2018-12-31",
    )
    assert train["sample_date"].tolist() == [pd.Timestamp("2016-12-16")]
    assert validation["sample_date"].tolist() == [pd.Timestamp("2017-01-06")]
    assert train["label_end_timestamp"].max() <= pd.Timestamp("2016-12-31")


def test_cross_sectional_target_matches_zscore_semantics() -> None:
    frame = pd.DataFrame(
        {
            "sample_date": [pd.Timestamp("2018-01-05")] * 4,
            "excess_return": [-0.2, 0.0, 0.1, 0.4],
        }
    )
    target = cross_sectional_zscore(frame, "excess_return")
    assert target.mean() == pytest.approx(0.0)
    assert target.std(ddof=1) == pytest.approx(1.0)
    assert target.is_monotonic_increasing


def test_rank_ic_and_decay_use_weekly_cross_sections() -> None:
    dates = pd.to_datetime(["2018-01-05", "2018-01-12"])
    rows = []
    labels = []
    for date in dates:
        for number in range(40):
            rows.append(
                {
                    "sample_date": date,
                    "instrument": f"S{number:03d}",
                    "prediction": float(number),
                    "excess_return": float(number),
                }
            )
            for horizon in [1, 5, 10, 20]:
                labels.append(
                    {
                        "sample_date": date,
                        "instrument": f"S{number:03d}",
                        "horizon": horizon,
                        "excess_return": float(number) / horizon,
                    }
                )
    frame = pd.DataFrame(rows)
    ic = rank_ic_series(frame, min_cross_section_size=30)
    assert ic["rank_ic"].tolist() == pytest.approx([1.0, 1.0])
    summary = summarize_ic(ic["rank_ic"])
    assert summary["rank_ic_mean"] == pytest.approx(1.0)
    decay = compute_ic_decay(
        frame[["sample_date", "instrument", "prediction"]],
        pd.DataFrame(labels),
        [1, 5, 10, 20],
        min_cross_section_size=30,
    )
    assert decay["rank_ic_mean"].tolist() == pytest.approx([1.0] * 4)


def test_decile_monotonicity_and_acceptance() -> None:
    rows = []
    for date in pd.to_datetime(["2018-01-05", "2018-01-12"]):
        for number in range(100):
            rows.append(
                {
                    "sample_date": date,
                    "instrument": f"S{number:03d}",
                    "prediction": float(number),
                    "excess_return": number / 1000.0,
                }
            )
    table, monotonicity = decile_return_table(pd.DataFrame(rows), deciles=10)
    assert len(table) == 10
    assert monotonicity["adjacent_up_steps"] == 9
    assert monotonicity["top_minus_bottom"] > 0
    acceptance = evaluate_acceptance(
        {"rank_ic_mean": 0.021, "icir": 0.251},
        monotonicity,
        rank_ic_mean_min=0.02,
        icir_min=0.25,
        monotonic_min_adjacent_steps=7,
    )
    assert acceptance["passed"]
    strict = evaluate_acceptance(
        {"rank_ic_mean": 0.02, "icir": 0.25},
        monotonicity,
        rank_ic_mean_min=0.02,
        icir_min=0.25,
        monotonic_min_adjacent_steps=7,
    )
    assert not strict["passed"]


def test_alpha158_contract_rejects_future_ref() -> None:
    validate_alpha158_spec(["Ref($close, 1)", "$open/$close"], ["A", "B"], 2)
    with pytest.raises(ValueError, match="未来 Ref"):
        validate_alpha158_spec(["Ref($close, -1)", "$open/$close"], ["A", "B"], 2)

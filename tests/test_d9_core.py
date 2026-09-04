from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pull_d9_oos import validate_annual_frame
from src.d9_core import (
    OOSIntegrityError,
    OOSThresholds,
    aggregate_daily_to_monthly,
    assert_frozen_config_unchanged,
    evaluate_oos_thresholds,
    maximum_relative_drawdown,
    newey_west_t_statistic,
    summarize_monthly_performance,
    validate_oos_fold_records,
)


def thresholds() -> OOSThresholds:
    return OOSThresholds(
        annualized_excess_return_min=0.035,
        information_ratio_min=0.75,
        newey_west_t_min=2.0,
        maximum_relative_drawdown_max=0.10,
        specific_contribution_share_min=0.50,
        doubled_impact_annualized_excess_min=0.0,
    )


def test_exact_eight_expanding_oos_folds_are_required() -> None:
    years = list(range(2019, 2027))
    frame = pd.DataFrame(
        {
            "prediction_year": years,
            "train_start": pd.Timestamp("2014-01-01"),
            "train_end": [pd.Timestamp(year - 1, 12, 31) for year in years],
            "prediction_start": [pd.Timestamp(year, 1, 1) for year in years],
            "prediction_end": [
                pd.Timestamp(year, 12, 31) if year < 2026 else pd.Timestamp("2026-08-31")
                for year in years
            ],
        }
    )
    audit = validate_oos_fold_records(
        frame, oos_start="2019-01-01", oos_end="2026-08-31"
    )
    assert audit["folds"] == 8
    assert audit["years"] == years
    broken = frame.loc[frame["prediction_year"].ne(2022)]
    with pytest.raises(OOSIntegrityError, match="完整运行"):
        validate_oos_fold_records(
            broken, oos_start="2019-01-01", oos_end="2026-08-31"
        )


def test_newey_west_t_matches_direct_bartlett_calculation() -> None:
    values = np.asarray([0.01, 0.02, -0.01, 0.03, 0.00, 0.02, 0.01, 0.04])
    lag = 2
    demeaned = values - values.mean()
    long_run = float(demeaned @ demeaned / len(values))
    for offset in range(1, lag + 1):
        covariance = float(
            demeaned[offset:] @ demeaned[:-offset] / len(values)
        )
        long_run += 2 * (1 - offset / (lag + 1)) * covariance
    expected = values.mean() / np.sqrt(long_run / len(values))
    assert newey_west_t_statistic(values, lag=lag) == pytest.approx(expected)
    with pytest.raises(ValueError, match="观察数"):
        newey_west_t_statistic(values[:3], lag=2)


def test_daily_returns_are_compounded_inside_each_month_before_excess() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"]
            ),
            "portfolio_return": [0.10, -0.05, 0.02, 0.03],
            "benchmark_return": [0.04, -0.02, 0.01, 0.01],
        }
    )
    monthly = aggregate_daily_to_monthly(frame)
    january = monthly.loc[monthly["month"].eq("2020-01")].iloc[0]
    assert january["portfolio_return"] == pytest.approx(1.10 * 0.95 - 1)
    assert january["benchmark_return"] == pytest.approx(1.04 * 0.98 - 1)
    assert january["excess_return"] == pytest.approx(
        january["portfolio_return"] - january["benchmark_return"]
    )


def test_monthly_ir_identity_and_conservative_nw_rule() -> None:
    dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    excess = np.tile([0.01, -0.002, 0.006, 0.003], 6)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "portfolio_return": excess,
            "benchmark_return": np.zeros(len(dates)),
        }
    )
    metrics, monthly = summarize_monthly_performance(
        daily, newey_west_lags=[3, 4, 5, 6]
    )
    assert len(monthly) == 24
    assert metrics["annualized_excess_return"] == pytest.approx(
        metrics["information_ratio"] * metrics["annualized_tracking_error"]
    )
    assert metrics["newey_west_t_minimum"] == pytest.approx(
        min(metrics["newey_west_t_by_lag"].values())
    )


def test_flat_zero_excess_has_zero_ir_and_t() -> None:
    dates = pd.date_range("2019-01-31", periods=12, freq="ME")
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "portfolio_return": np.zeros(len(dates)),
            "benchmark_return": np.zeros(len(dates)),
        }
    )
    metrics, _ = summarize_monthly_performance(daily, newey_west_lags=[3])
    assert metrics["information_ratio"] == 0.0
    assert metrics["newey_west_t_minimum"] == 0.0
    assert metrics["identity_ir_times_te"] == 0.0


def test_maximum_relative_drawdown_uses_relative_wealth() -> None:
    daily = pd.DataFrame(
        {
            "portfolio_return": [0.10, -0.20, 0.05],
            "benchmark_return": [0.0, 0.0, 0.0],
        }
    )
    # Relative wealth: 1.10 -> 0.88 -> 0.924, so the maximum drawdown is 20%.
    assert maximum_relative_drawdown(daily) == pytest.approx(0.20)


def test_maximum_relative_drawdown_includes_initial_wealth_peak() -> None:
    daily = pd.DataFrame(
        {
            "portfolio_return": [-0.05, 0.01],
            "benchmark_return": [0.0, 0.0],
        }
    )
    assert maximum_relative_drawdown(daily) == pytest.approx(0.05)


def test_three_state_decision_preserves_d10_attribution_dependency() -> None:
    strong = {
        "annualized_excess_return": 0.05,
        "information_ratio": 1.0,
        "newey_west_t_minimum": 2.5,
        "maximum_relative_drawdown": 0.08,
    }
    pending = evaluate_oos_thresholds(
        strong,
        doubled_impact_annualized_excess=0.01,
        thresholds=thresholds(),
    )
    assert pending["decision"] == "待D10归因"
    assert pending["checks"]["specific_contribution_share"] is None
    go = evaluate_oos_thresholds(
        strong,
        doubled_impact_annualized_excess=0.01,
        thresholds=thresholds(),
        specific_contribution_share=0.60,
    )
    assert go["decision"] == "Go"
    weak_t = dict(strong, newey_west_t_minimum=1.9)
    assert evaluate_oos_thresholds(
        weak_t,
        doubled_impact_annualized_excess=0.01,
        thresholds=thresholds(),
    )["decision"] == "证据不足"
    weak_ir = dict(strong, information_ratio=0.70)
    assert evaluate_oos_thresholds(
        weak_ir,
        doubled_impact_annualized_excess=0.01,
        thresholds=thresholds(),
    )["decision"] == "No-Go"


def test_only_new_d9_config_section_may_differ_from_freeze() -> None:
    frozen = {"windows": {"design_end": "2018-12-31"}, "d8": {"top_n": 100}}
    current = {**frozen, "d9": {"expected_fold_count": 8}}
    assert_frozen_config_unchanged(current, frozen)
    changed = {**current, "d8": {"top_n": 99}}
    with pytest.raises(OOSIntegrityError, match="frozen-oos"):
        assert_frozen_config_unchanged(changed, frozen)


def test_d9_ingestion_rejects_missing_required_market_date() -> None:
    expected = pd.DatetimeIndex(["2019-01-02", "2019-01-03"])
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20190102"],
            "close": [10.0],
            "turnover_rate": [1.0],
            "turnover_rate_f": [1.0],
            "volume_ratio": [1.0],
            "pe": [10.0],
            "pe_ttm": [10.0],
            "pb": [1.0],
            "ps": [1.0],
            "ps_ttm": [1.0],
            "dv_ratio": [0.0],
            "dv_ttm": [0.0],
            "total_share": [1.0],
            "float_share": [1.0],
            "free_share": [1.0],
            "total_mv": [10.0],
            "circ_mv": [10.0],
        }
    )
    with pytest.raises(ValueError, match="缺少 1 个交易日"):
        validate_annual_frame(
            frame,
            endpoint="daily_basic",
            year=2019,
            expected_dates=expected,
        )


def test_d9_ingestion_allows_empty_suspension_year() -> None:
    frame = pd.DataFrame(
        columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
    )
    audit = validate_annual_frame(
        frame,
        endpoint="suspend_d",
        year=2019,
        expected_dates=pd.DatetimeIndex(["2019-01-02"]),
    )
    assert audit["passed"] is True
    assert audit["rows"] == 0

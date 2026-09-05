from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pull_d9_oos import validate_annual_frame
from src.d9_core import (
    OOSIntegrityError,
    OOSThresholds,
    aggregate_daily_to_monthly,
    apply_identifier_alias_removals,
    apply_snapshot_instrument_bridges,
    assert_frozen_config_unchanged,
    evaluate_oos_thresholds,
    maximum_relative_drawdown,
    newey_west_t_statistic,
    summarize_monthly_performance,
    validate_membership_recovery_failure,
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


def test_identifier_alias_removal_keeps_canonical_and_splits_alias() -> None:
    intervals = pd.DataFrame(
        {
            "instrument": ["SZ300114", "SZ302132"],
            "start_date": ["2023-01-01", "2023-12-29"],
            "end_date": ["2025-12-31", "2025-06-29"],
        }
    )
    result, audit = apply_identifier_alias_removals(
        intervals,
        [
            {
                "effective_start": "2023-12-29",
                "effective_end": "2025-06-29",
                "alias_instrument": "SZ300114",
                "canonical_instrument": "SZ302132",
            }
        ],
    )
    alias = result.loc[result["instrument"].eq("SZ300114")]
    assert alias[["start_date", "end_date"]].to_dict(orient="records") == [
        {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-28"),
        },
        {
            "start_date": pd.Timestamp("2025-06-30"),
            "end_date": pd.Timestamp("2025-12-31"),
        },
    ]
    assert len(audit) == 1


def test_identifier_alias_removal_requires_canonical_coverage() -> None:
    intervals = pd.DataFrame(
        {
            "instrument": ["SZ300114", "SZ302132"],
            "start_date": ["2023-01-01", "2024-01-01"],
            "end_date": ["2025-12-31", "2025-06-29"],
        }
    )
    with pytest.raises(ValueError, match="未完整覆盖"):
        apply_identifier_alias_removals(
            intervals,
            [
                {
                    "effective_start": "2023-12-29",
                    "effective_end": "2025-06-29",
                    "alias_instrument": "SZ300114",
                    "canonical_instrument": "SZ302132",
                }
            ],
        )


def test_synthetic_snapshot_bridge_preserves_weight_and_raw_snapshots() -> None:
    snapshots = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2021-09-13", "2021-09-13"]),
            "instrument": ["SH600297", "SH601868"],
            "weight": [0.8, 0.2],
            "snapshot_origin": [
                "backcast_membership_correction",
                "backcast_membership_correction",
            ],
        }
    )
    result, audit = apply_snapshot_instrument_bridges(
        snapshots,
        [
            {
                "trade_date": "2021-09-13",
                "source_instrument": "SH601868",
                "target_instrument": "SH600068",
            }
        ],
    )
    bridged = result.set_index("instrument")["weight"]
    assert "SH601868" not in bridged.index
    assert bridged["SH600068"] == pytest.approx(0.2)
    assert audit[0]["weight_preserved"] is True
    raw = snapshots.assign(snapshot_origin="tushare_index_weight")
    with pytest.raises(ValueError, match="不得修改"):
        apply_snapshot_instrument_bridges(
            raw,
            [
                {
                    "trade_date": "2021-09-13",
                    "source_instrument": "SH601868",
                    "target_instrument": "SH600068",
                }
            ],
        )


def test_membership_recovery_accepts_only_exact_registered_failure() -> None:
    expected = {
        "status": "failed",
        "oos_run_number": 1,
        "d9_head_commit": "b479886",
        "config_sha256": "old-config",
        "error_type": "ValueError",
        "error": "D9 权重快照与 PIT 成分区间有 35 个不一致日",
        "rerun_allowed": False,
    }
    manifest = {
        **expected,
        "traceback": "Traceback ... in build_full_benchmark_weights",
    }
    audit = validate_membership_recovery_failure(manifest, expected)
    assert audit["validated"] is True
    assert audit["performance_observed_before_failure"] is False
    changed = {**manifest, "error": "another failure"}
    with pytest.raises(OOSIntegrityError, match="不一致"):
        validate_membership_recovery_failure(changed, expected)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.d5_pipeline import qlib_amount_multiplier
from src.d5_core import (
    CalibrationError,
    ExpandingFold,
    apply_return_calibration,
    assert_fold_integrity,
    build_annual_calibration_schedule,
    build_expanding_year_folds,
    calibration_scale_diagnostics,
    fit_return_calibration,
    rank_gaussianize,
    split_walk_forward_samples,
)


def test_qlib_amount_multiplier_uses_tradability_config_section() -> None:
    config = {
        "tradability": {"qlib_amount_multiplier_to_cny": 1000},
        "validation": {},
    }
    assert qlib_amount_multiplier(config) == 1000.0
    with pytest.raises(ValueError, match="tradability.qlib_amount_multiplier_to_cny"):
        qlib_amount_multiplier({"validation": {}})


def test_build_expanding_folds_uses_calendar_years_and_partial_last_year() -> None:
    folds = build_expanding_year_folds(
        train_start="2014-01-01",
        initial_train_end="2018-12-31",
        prediction_start="2019-01-01",
        prediction_end="2026-08-31",
    )
    assert len(folds) == 8
    assert folds[0].prediction_year == 2019
    assert folds[0].train_end == pd.Timestamp("2018-12-31")
    assert folds[-1].prediction_year == 2026
    assert folds[-1].train_end == pd.Timestamp("2025-12-31")
    assert folds[-1].prediction_end == pd.Timestamp("2026-08-31")
    assert {fold.train_start for fold in folds} == {pd.Timestamp("2014-01-01")}


def test_fold_dates_must_be_contiguous_at_first_prediction_boundary() -> None:
    with pytest.raises(ValueError, match="必须紧邻"):
        build_expanding_year_folds(
            train_start="2014-01-01",
            initial_train_end="2016-12-30",
            prediction_start="2017-01-01",
            prediction_end="2018-12-31",
        )


def test_walk_forward_split_purges_labels_crossing_cutoffs() -> None:
    fold = ExpandingFold(
        prediction_year=2017,
        train_start=pd.Timestamp("2014-01-01"),
        train_end=pd.Timestamp("2016-12-31"),
        prediction_start=pd.Timestamp("2017-01-01"),
        prediction_end=pd.Timestamp("2017-12-31"),
    )
    frame = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(
                ["2016-12-16", "2016-12-23", "2017-01-06", "2017-12-22"]
            ),
            "label_end_timestamp": pd.to_datetime(
                ["2016-12-23", "2017-01-03", "2017-01-13", "2018-01-03"]
            ),
            "instrument": ["A"] * 4,
        }
    )
    train, prediction = split_walk_forward_samples(frame, fold)
    assert train["sample_date"].tolist() == [pd.Timestamp("2016-12-16")]
    assert prediction["sample_date"].tolist() == [pd.Timestamp("2017-01-06")]
    assert train["label_end_timestamp"].max() <= fold.train_end


def test_fold_integrity_rejects_training_label_after_cutoff() -> None:
    fold = ExpandingFold(
        prediction_year=2017,
        train_start=pd.Timestamp("2014-01-01"),
        train_end=pd.Timestamp("2016-12-31"),
        prediction_start=pd.Timestamp("2017-01-01"),
        prediction_end=pd.Timestamp("2017-12-31"),
    )
    train = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(["2016-12-23"]),
            "label_end_timestamp": pd.to_datetime(["2017-01-03"]),
        }
    )
    prediction = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(["2017-01-06"]),
            "label_end_timestamp": pd.to_datetime(["2017-01-13"]),
        }
    )
    with pytest.raises(AssertionError, match="标签结束日"):
        assert_fold_integrity(train, prediction, fold)


def test_rank_gaussianize_is_order_preserving_and_standardized_per_date() -> None:
    dates = pd.to_datetime(["2017-01-06", "2017-01-13"])
    frame = pd.DataFrame(
        {
            "sample_date": np.repeat(dates, 8),
            "prediction": np.concatenate([np.arange(8), np.arange(8)[::-1]]),
        }
    )
    result = rank_gaussianize(frame)
    for date, index in frame.groupby("sample_date").groups.items():
        group = result.loc[index]
        assert group.mean() == pytest.approx(0.0, abs=1e-12)
        assert group.std(ddof=1) == pytest.approx(1.0, abs=1e-12)
        ordered = frame.loc[index].assign(score_z=group).sort_values("prediction")
        assert ordered["score_z"].is_monotonic_increasing


def test_rank_gaussianize_rejects_constant_cross_section_by_nan() -> None:
    frame = pd.DataFrame(
        {
            "sample_date": [pd.Timestamp("2017-01-06")] * 4,
            "prediction": [1.0] * 4,
        }
    )
    assert rank_gaussianize(frame).isna().all()


def test_zero_intercept_calibration_recovers_expected_return_scale() -> None:
    frame = pd.DataFrame(
        {
            "sample_date": np.repeat(pd.to_datetime(["2017-01-06", "2017-01-13"]), 4),
            "score_z": np.tile([-1.5, -0.5, 0.5, 1.5], 2),
        }
    )
    frame["excess_return"] = 0.004 * frame["score_z"]
    result = fit_return_calibration(frame)
    assert result["coefficient"] == pytest.approx(0.004)
    calibrated = apply_return_calibration(frame["score_z"], result["coefficient"])
    assert calibrated.to_numpy() == pytest.approx(frame["excess_return"].to_numpy())


def test_calibration_scale_matches_realized_decile_curve() -> None:
    rows = []
    for date in pd.date_range("2017-01-06", periods=4, freq="7D"):
        for number in range(100):
            rows.append(
                {
                    "sample_date": date,
                    "instrument": f"S{number:03d}",
                    "prediction": float(number),
                }
            )
    frame = pd.DataFrame(rows)
    frame["score_z"] = rank_gaussianize(frame)
    frame["excess_return"] = 0.003 * frame["score_z"]
    calibration = fit_return_calibration(frame)
    diagnostics, curve = calibration_scale_diagnostics(
        frame,
        calibration["coefficient"],
        deciles=10,
        tolerance=0.30,
    )
    assert calibration["coefficient"] == pytest.approx(0.003)
    assert len(curve) == 10
    assert diagnostics["passed"]
    assert 0.7 <= diagnostics["scale_ratio"] <= 1.3


def test_annual_calibration_schedule_uses_only_prior_years() -> None:
    rows = []
    for year, slope in [(2017, 0.002), (2018, 0.004)]:
        for week in [1, 2]:
            date = pd.Timestamp(year=year, month=1, day=week * 7)
            for score in [-1.0, 1.0]:
                rows.append(
                    {
                        "sample_date": date,
                        "score_z": score,
                        "excess_return": slope * score,
                    }
                )
    schedule = build_annual_calibration_schedule(pd.DataFrame(rows), [2018, 2019])
    first = schedule.set_index("effective_prediction_year").loc[2018]
    second = schedule.set_index("effective_prediction_year").loc[2019]
    assert first["history_end"].startswith("2017-")
    assert first["calibration_coefficient"] == pytest.approx(0.002)
    assert second["history_end"].startswith("2018-")
    assert second["calibration_coefficient"] == pytest.approx(0.003)
    assert bool(schedule["history_precedes_effective_year"].all())


def test_annual_calibration_schedule_requires_prior_history() -> None:
    frame = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(["2017-01-06", "2017-01-06"]),
            "score_z": [-1.0, 1.0],
            "excess_return": [-0.002, 0.002],
        }
    )
    with pytest.raises(CalibrationError, match="校准历史"):
        build_annual_calibration_schedule(frame, [2017])

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.d2_core import (
    add_backcast_snapshots,
    apply_membership_replacements,
    build_daily_weights,
    classify_limit_quality,
    expand_membership,
    make_crosscheck,
    to_qlib_symbol,
    to_tushare_symbol,
)
from src.data_access import ResearchDataAccess, WindowViolation


def test_symbol_roundtrip() -> None:
    assert to_qlib_symbol("600000.SH") == "SH600000"
    assert to_tushare_symbol("SZ000001") == "000001.SZ"


def test_membership_and_return_drift() -> None:
    calendar = pd.date_range("2018-01-02", periods=3, freq="D")
    intervals = pd.DataFrame(
        {
            "instrument": ["SH600000", "SZ000001"],
            "start_date": [calendar[0], calendar[0]],
            "end_date": [calendar[-1], calendar[-1]],
        }
    )
    membership = expand_membership(intervals, calendar)
    snapshots = pd.DataFrame(
        {
            "trade_date": [calendar[0], calendar[0]],
            "instrument": ["SH600000", "SZ000001"],
            "weight": [60.0, 40.0],
        }
    )
    close = pd.DataFrame(
        {
            "SH600000": [10.0, 11.0, 11.0],
            "SZ000001": [20.0, 20.0, 22.0],
        },
        index=calendar,
    )
    result = build_daily_weights(
        calendar, membership, snapshots, close, calendar[0], calendar[-1]
    )
    sums = result.groupby("trade_date")["weight"].sum()
    assert np.allclose(sums, 1.0)
    day_two = result.loc[result["trade_date"].eq(calendar[1])].set_index("instrument")
    assert day_two.loc["SH600000", "weight"] == pytest.approx(0.66 / 1.06)


def test_membership_replacement_is_inclusive_and_audited() -> None:
    intervals = pd.DataFrame(
        {
            "instrument": ["SH600549", "SH600848"],
            "start_date": ["2017-01-26", "2017-06-30"],
            "end_date": ["2017-06-29", "2017-12-28"],
        }
    )
    replacements = [
        {
            "effective_start": "2017-02-14",
            "effective_end": "2017-06-29",
            "remove_instrument": "SH600549",
            "add_instrument": "SH600848",
            "anchor_snapshot_date": "2017-02-28",
            "reason": "test",
        }
    ]
    corrected, audit = apply_membership_replacements(intervals, replacements)
    calendar = pd.to_datetime(
        ["2017-02-13", "2017-02-14", "2017-06-29", "2017-06-30"]
    )
    membership = expand_membership(corrected, calendar)
    sets = membership.groupby("trade_date")["instrument"].agg(set)
    assert sets.loc[pd.Timestamp("2017-02-13")] == {"SH600549"}
    assert sets.loc[pd.Timestamp("2017-02-14")] == {"SH600848"}
    assert sets.loc[pd.Timestamp("2017-06-29")] == {"SH600848"}
    assert sets.loc[pd.Timestamp("2017-06-30")] == {"SH600848"}
    assert len(audit) == 1


def test_backcast_snapshot_reproduces_anchor_weights() -> None:
    dates = pd.to_datetime(["2017-02-14", "2017-02-28"])
    snapshots = pd.DataFrame(
        {
            "trade_date": [dates[1], dates[1]],
            "instrument": ["SH600001", "SH600002"],
            "weight": [0.6, 0.4],
        }
    )
    close = pd.DataFrame(
        {"SH600001": [10.0, 12.0], "SH600002": [20.0, 18.0]},
        index=dates,
    )
    replacements = [
        {
            "effective_start": "2017-02-14",
            "effective_end": "2017-06-29",
            "remove_instrument": "SH600000",
            "add_instrument": "SH600002",
            "anchor_snapshot_date": "2017-02-28",
        }
    ]
    result = add_backcast_snapshots(snapshots, close, replacements)
    effective = result.loc[result["trade_date"].eq(dates[0])].set_index(
        "instrument"
    )["weight"]
    drifted = effective * pd.Series({"SH600001": 1.2, "SH600002": 0.9})
    drifted = drifted / drifted.sum()
    assert drifted.loc["SH600001"] == pytest.approx(0.6)
    assert drifted.loc["SH600002"] == pytest.approx(0.4)


def test_limit_quality_accepts_only_complete_zero_placeholders() -> None:
    limits = pd.DataFrame(
        {
            "pre_close": [0.0, 10.0, 10.0, 10.0],
            "up_limit": [0.0, 11.0, 0.0, 9.0],
            "down_limit": [0.0, 9.0, 9.0, 10.0],
        }
    )
    placeholders, anomalies = classify_limit_quality(limits)
    assert placeholders.tolist() == [True, False, False, False]
    assert anomalies.tolist() == [False, False, True, True]


def test_crosscheck_uses_raw_price_and_relative_factor() -> None:
    dates = pd.to_datetime(["2018-01-02", "2018-06-01"])
    qlib_frame = pd.DataFrame(
        {
            "instrument": ["SH600000", "SH600000"],
            "trade_date": dates,
            "close": [15.0, 33.0],
            "factor": [1.5, 3.0],
        }
    )
    daily = pd.DataFrame(
        {"ts_code": ["600000.SH"] * 2, "trade_date": dates, "close": [10.0, 11.0]}
    )
    factor = pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * 2,
            "trade_date": dates,
            "adj_factor": [2.0, 4.0],
        }
    )
    result = make_crosscheck(qlib_frame, daily, factor)
    assert result["raw_close_relative_error"].max() == pytest.approx(0.0)
    assert result["factor_ratio_relative_error"].max() == pytest.approx(0.0)


def test_design_guard_blocks_2019(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    config = {
        "data": {
            "window_marker": "data/.window_mode.json",
            "qlib_provider_uri": "/not/used",
        },
        "windows": {"design_end": "2018-12-31"},
    }
    (tmp_path / "config/config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    marker = tmp_path / "data/.window_mode.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"mode": "design"}), encoding="utf-8")
    access = ResearchDataAccess(tmp_path)
    access.assert_window("2014-01-01", "2018-12-31")
    with pytest.raises(WindowViolation):
        access.assert_window("2014-01-01", "2019-01-01")

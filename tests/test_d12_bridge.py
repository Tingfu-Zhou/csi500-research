"""Day-2 bridge localisation: the arithmetic, and the refusal to decompose what it cannot reproduce.

These run on synthetic frames and need no Qlib data. The identity they pin is the one in
`src.d11_diagnostics.attribute_d11`: the basket drops a constituent whose return is
unobservable rather than treating it as a zero, and whatever the index earned on that name
stays in the bridge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_bridge import (basket_contributions, bridge_by_day, membership_timeline,
                            month_share_of_bridge, reproduction_verdict, summarise_bridge)
from src.d12_repo import D12Error


def weight_panel(dates, members_by_date, snapshot_by_date=None):
    rows = []
    for date in dates:
        members = members_by_date[date]
        for name in members:
            rows.append({"trade_date": pd.Timestamp(date), "instrument": name,
                         "weight": 1 / len(members),
                         "snapshot_date": pd.Timestamp((snapshot_by_date or {}).get(date, date))})
    return pd.DataFrame(rows)


def test_membership_timeline_reports_who_joined_and_left():
    dates = ["2024-06-10", "2024-06-11", "2024-06-12"]
    members = {dates[0]: ["A", "B", "C"], dates[1]: ["A", "B", "C"], dates[2]: ["A", "B", "D"]}
    timeline = membership_timeline(weight_panel(dates, members)).set_index("trade_date")
    first, quiet, change = (timeline.loc[pd.Timestamp(d)] for d in dates)
    assert pd.isna(first["weight_l1_change"]) and first["members"] == 3
    assert quiet["added"] == 0 and quiet["removed"] == 0 and not quiet["membership_changed"]
    assert quiet["weight_l1_change"] == pytest.approx(0.0)
    assert change["added"] == 1 and change["removed"] == 1 and change["membership_changed"]
    assert change["added_weight"] == pytest.approx(1 / 3)
    assert change["removed_weight"] == pytest.approx(1 / 3)
    assert change["weight_l1_change"] == pytest.approx(2 / 3)
    assert np.allclose(timeline["weight_sum"], 1.0)


def test_membership_timeline_measures_snapshot_staleness_in_trade_days():
    dates = ["2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13"]
    members = {date: ["A", "B"] for date in dates}
    snapshots = {dates[0]: dates[0], dates[1]: dates[0], dates[2]: dates[0], dates[3]: dates[3]}
    timeline = membership_timeline(weight_panel(dates, members, snapshots))
    assert list(timeline["days_since_snapshot"]) == [0, 1, 2, 0]
    assert list(timeline["is_snapshot_day"]) == [True, False, False, True]


def test_membership_timeline_rejects_ambiguous_input():
    dates = ["2024-06-10"]
    panel = weight_panel(dates, {dates[0]: ["A", "B"]})
    with pytest.raises(D12Error, match="duplicate"):
        membership_timeline(pd.concat([panel, panel], ignore_index=True))
    conflicting = panel.copy()
    conflicting.loc[0, "snapshot_date"] = pd.Timestamp("2024-05-31")
    with pytest.raises(D12Error, match="more than one weight snapshot"):
        membership_timeline(conflicting)
    with pytest.raises(D12Error, match="missing"):
        membership_timeline(panel.drop(columns=["weight"]))


def attribution_panel(bridges, exposure_lag=1):
    dates = pd.bdate_range("2024-06-03", periods=len(bridges))
    return pd.DataFrame({
        "trade_date": dates,
        "exposure_date": dates - pd.tseries.offsets.BDay(exposure_lag),
        "benchmark_bridge": bridges,
        "benchmark_return": np.linspace(-0.01, 0.01, len(bridges)),
        "unobservable_benchmark_weight": 0.0})


def test_bridge_by_day_joins_on_the_weight_vector_that_priced_the_day():
    """attribute_d11 uses the previous day's weights, so the join must use exposure_date."""
    bridges = [0.0001, -0.012, 0.0002]
    attribution = attribution_panel(bridges)
    exposure_dates = [d.strftime("%Y-%m-%d") for d in attribution["exposure_date"]]
    members = {exposure_dates[0]: ["A", "B"], exposure_dates[1]: ["A", "B"],
               exposure_dates[2]: ["A", "C"]}
    timeline = membership_timeline(weight_panel(exposure_dates, members))
    frame = bridge_by_day(attribution, timeline)
    assert list(frame["abs_bridge"]) == [abs(b) for b in bridges]
    # The change happened on the third exposure date, so it lands on the third trade date.
    assert list(frame["membership_changed"]) == [False, False, True]
    assert frame["members"].tolist() == [2, 2, 2]


def test_bridge_by_day_filters_months_and_refuses_an_empty_scope():
    attribution = attribution_panel([0.001] * 5)
    assert len(bridge_by_day(attribution, months=["2024-06"])) == 5
    with pytest.raises(D12Error, match="no attribution rows"):
        bridge_by_day(attribution, months=["2019-01"])
    with pytest.raises(D12Error, match="missing"):
        bridge_by_day(attribution.drop(columns=["benchmark_bridge"]))


def test_month_share_shows_a_single_day_carrying_the_month():
    attribution = attribution_panel([1e-5, 1e-5, -0.015, 1e-5, 1e-5])
    frame = bridge_by_day(attribution)
    per_month = month_share_of_bridge(frame, top=1).iloc[0]
    assert per_month["month"] == "2024-06"
    assert per_month["share_in_top_1_days"] > 0.99
    assert per_month["worst_day_bridge"] == pytest.approx(-0.015)


def test_summarise_contrasts_rebalance_months_with_the_rest():
    june = attribution_panel([0.01] * 4)
    july = attribution_panel([0.001] * 4)
    july["trade_date"] = july["trade_date"] + pd.Timedelta(days=30)
    july["exposure_date"] = july["exposure_date"] + pd.Timedelta(days=30)
    frame = bridge_by_day(pd.concat([june, july], ignore_index=True))
    summary = summarise_bridge(frame)
    rebalance = summary["rebalance_months"]
    assert rebalance["days"] == 4 and rebalance["share_of_days"] == pytest.approx(0.5)
    assert rebalance["share_of_absolute_bridge"] == pytest.approx(10 / 11)
    ratio = rebalance["mean_absolute_bridge_inside"] / rebalance["mean_absolute_bridge_outside"]
    assert ratio == pytest.approx(10.0)


# --- the basket side, and only the basket side --------------------------------------

def test_basket_contributions_reproduce_the_attribution_basket_return():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    returns = pd.Series({"A": 0.02, "B": -0.01, "C": 0.005})
    table, summary = basket_contributions(weights, returns)
    assert summary["basket_return"] == pytest.approx(float(weights @ returns))
    assert summary["unobservable_weight"] == 0.0
    assert table["contribution"].sum() == pytest.approx(summary["basket_return"])


def test_an_unpriced_constituent_is_dropped_not_zeroed():
    """attribute_d11 removes it from the basket; its index effect stays in the bridge."""
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    returns = pd.Series({"A": 0.02, "B": np.nan, "C": 0.005})
    table, summary = basket_contributions(weights, returns)
    assert summary["basket_return"] == pytest.approx(0.5 * 0.02 + 0.2 * 0.005)
    assert summary["unobservable_weight"] == pytest.approx(0.3)
    assert summary["unobservable_names"] == ["B"]
    assert bool(table.loc["B", "observable"]) is False
    assert pd.isna(table.loc["B", "contribution"])
    # Zeroing it instead would have produced the same number here, so pin the difference
    # on a day where the dropped name would have mattered.
    assert summary["basket_return"] != pytest.approx(float((weights * returns.fillna(0.05)).sum()))


def test_basket_contributions_sum_to_the_bridge_plus_the_official_return():
    """bridge = basket - official, so the decomposition closes only with the index number."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.03, "B": -0.01})
    _, summary = basket_contributions(weights, returns)
    official = 0.012
    bridge = summary["basket_return"] - official
    assert summary["basket_return"] == pytest.approx(bridge + official)
    assert bridge == pytest.approx(0.6 * 0.03 + 0.4 * -0.01 - official)


# --- the gate that decides whether a decomposition may be shown at all ---------------

def test_reproduction_gate_accepts_float32_noise_and_rejects_a_real_gap():
    """Qlib stores prices as float32, so a faithful rebuild lands ~1e-7 relative away."""
    saved = 0.01059713
    float32_noise = reproduction_verdict(saved + 1.4e-09, saved)
    assert float32_noise["reproduced"]
    assert float32_noise["relative_error"] == pytest.approx(1.4e-09 / saved, rel=1e-6)
    # A gap big enough to reorder basis-point contributions must still fail.
    assert not reproduction_verdict(saved + 1e-05, saved)["reproduced"]
    # The absolute arm still governs when the saved value is ~0.
    assert reproduction_verdict(1e-11, 0.0)["reproduced"]
    assert not reproduction_verdict(1e-08, 0.0)["reproduced"]


def test_reproduction_gate_is_the_larger_of_the_two_arms():
    verdict = reproduction_verdict(1.0, 1.0, absolute_tolerance=1e-3, relative_tolerance=1e-9)
    assert verdict["tolerance"] == pytest.approx(1e-3)
    verdict = reproduction_verdict(1.0, 1.0, absolute_tolerance=1e-12, relative_tolerance=1e-6)
    assert verdict["tolerance"] == pytest.approx(1e-6)
    # Loosening is never silent: the error and the gate both travel with the verdict.
    assert set(verdict) == {"rebuilt", "saved", "error", "relative_error", "tolerance", "reproduced"}

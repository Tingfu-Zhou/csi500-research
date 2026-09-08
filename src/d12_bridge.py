"""Locate the benchmark bridge: month -> day -> security.

`src.d11_diagnostics.attribute_d11` defines, for each day,

    bridge = sum_i wb_i * r_i   (observable i only)   -   official_index_return

where `wb` is the benchmark weight vector as of the PREVIOUS trading day (the row's
`exposure_date`) and `r` is that day's close-to-close return. Only the basket side
decomposes by security: the official index is one published number, and the plan forbids
inventing a security-level difference against it. So this module reports the basket
contributions it can attribute and leaves the index-level gap standing as a gap.

The bridge is not spread evenly. Over the 92 saved months its mean absolute size is 9.5x
larger in June and December than in the other ten months, and 10 of the 12 largest months
fall in those two — which is when the CSI 500 reconstitutes. `membership_timeline` and
`summarise_bridge` exist to test that: if reconstitution is the mechanism, the month's
bridge should sit on the days the constituent set changes, not spread across the month.

Nothing here reads a label or a D11 output other than the saved attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.d12_repo import D12Error

WEIGHT_COLUMNS = ("trade_date", "instrument", "weight")
BRIDGE_COLUMNS = ("trade_date", "exposure_date", "benchmark_bridge", "benchmark_return")


def _require(frame, columns, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise D12Error(f"{name} is missing {missing}")


def membership_timeline(weights):
    """Per trade_date: who joined, who left, and how far the weights moved.

    `snapshot_date` comes from `src.d2_core.build_daily_weights`: it records which
    Tushare snapshot the day's weights descend from, so `days_since_snapshot` measures
    how long the reconstructed basket has been drifting on stale published weights.
    """
    _require(weights, WEIGHT_COLUMNS, "benchmark weights")
    frame = weights.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise D12Error("benchmark weights have duplicate (trade_date, instrument) rows")
    has_snapshot = "snapshot_date" in frame.columns
    if has_snapshot:
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])

    groups = {date: group for date, group in frame.groupby("trade_date", sort=True)}
    dates = sorted(groups)
    rows, previous = [], None
    for date in dates:
        group = groups[date]
        current = group.set_index("instrument")["weight"].astype(float)
        row = {"trade_date": date, "members": int(len(current)),
               "weight_sum": float(current.sum())}
        if previous is None:
            row.update(added=0, removed=0, added_weight=np.nan, removed_weight=np.nan,
                       weight_l1_change=np.nan, membership_changed=False)
        else:
            added = current.index.difference(previous.index)
            removed = previous.index.difference(current.index)
            union = current.index.union(previous.index)
            moved = (current.reindex(union, fill_value=0.0)
                     - previous.reindex(union, fill_value=0.0)).abs().sum()
            row.update(added=int(len(added)), removed=int(len(removed)),
                       added_weight=float(current.reindex(added).sum()),
                       removed_weight=float(previous.reindex(removed).sum()),
                       weight_l1_change=float(moved),
                       membership_changed=bool(len(added) or len(removed)))
        if has_snapshot:
            snapshot = group["snapshot_date"].dropna().unique()
            if len(snapshot) > 1:
                raise D12Error(f"{date.date()}: more than one weight snapshot on one day")
            row["snapshot_date"] = pd.Timestamp(snapshot[0]) if len(snapshot) else pd.NaT
        rows.append(row)
        previous = current

    timeline = pd.DataFrame(rows)
    if has_snapshot:
        position = pd.Series(range(len(dates)), index=pd.DatetimeIndex(dates))
        timeline["days_since_snapshot"] = (
            timeline["trade_date"].map(position)
            - timeline["snapshot_date"].map(position)).astype("Float64")
        timeline["is_snapshot_day"] = timeline["snapshot_date"].eq(timeline["trade_date"])
    return timeline


def bridge_by_day(attribution_daily, timeline=None, months=None):
    """Daily bridge, joined to the weight vector that actually produced it.

    The join is on `exposure_date`, not `trade_date`: `attribute_d11` prices the day's
    returns with the previous day's weights, so that is the vector whose staleness and
    membership matter. A reconstitution effective at the open of day D therefore meets a
    basket still holding day D-1's constituents — a one-day mismatch by construction, and
    the first thing to check when a month's bridge turns out to sit on a single day.
    """
    _require(attribution_daily, BRIDGE_COLUMNS, "attribution_daily")
    frame = attribution_daily.loc[:, list(BRIDGE_COLUMNS)].copy()
    for column in ("trade_date", "exposure_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    if "unobservable_benchmark_weight" in attribution_daily.columns:
        frame["unobservable_benchmark_weight"] = attribution_daily["unobservable_benchmark_weight"]
    frame["month"] = frame["trade_date"].dt.strftime("%Y-%m")
    if months:
        frame = frame.loc[frame["month"].isin(set(months))]
        if frame.empty:
            raise D12Error(f"no attribution rows in {sorted(set(months))}")
    frame["abs_bridge"] = frame["benchmark_bridge"].abs()
    if timeline is not None:
        frame = frame.merge(timeline.rename(columns={"trade_date": "exposure_date"}),
                            on="exposure_date", how="left", suffixes=("", "_weights"))
    return frame.sort_values("trade_date").reset_index(drop=True)


def month_share_of_bridge(frame, top=5):
    """For each month: how much of its absolute bridge sits on its worst few days."""
    output = []
    for month, group in frame.groupby("month", sort=True):
        ordered = group.sort_values("abs_bridge", ascending=False)
        total = float(ordered["abs_bridge"].sum())
        head = ordered.head(top)
        output.append({
            "month": month, "days": int(len(group)),
            "signed_bridge": float(group["benchmark_bridge"].sum()),
            "absolute_bridge": total,
            f"share_in_top_{top}_days": float(head["abs_bridge"].sum() / total) if total else np.nan,
            "worst_day": str(ordered.iloc[0]["trade_date"].date()),
            "worst_day_bridge": float(ordered.iloc[0]["benchmark_bridge"]),
            "worst_day_membership_changed":
                bool(ordered.iloc[0]["membership_changed"])
                if "membership_changed" in ordered.columns else None})
    return pd.DataFrame(output)


def summarise_bridge(frame, rebalance_months=("06", "12")):
    """Does the bridge behave like a reconstitution effect, or like a spread-out drift?"""
    total = float(frame["abs_bridge"].sum())
    summary = {"days": int(len(frame)), "absolute_bridge_total": total,
               "signed_bridge_total": float(frame["benchmark_bridge"].sum()),
               "mean_absolute_bridge": float(frame["abs_bridge"].mean())}
    calendar = frame["trade_date"].dt.strftime("%m")
    in_rebalance = calendar.isin(set(rebalance_months))
    summary["rebalance_months"] = {
        "months": list(rebalance_months),
        "days": int(in_rebalance.sum()),
        "share_of_days": float(in_rebalance.mean()),
        "share_of_absolute_bridge": float(frame.loc[in_rebalance, "abs_bridge"].sum() / total)
        if total else np.nan,
        "mean_absolute_bridge_inside": float(frame.loc[in_rebalance, "abs_bridge"].mean())
        if in_rebalance.any() else np.nan,
        "mean_absolute_bridge_outside": float(frame.loc[~in_rebalance, "abs_bridge"].mean())
        if (~in_rebalance).any() else np.nan}
    if "membership_changed" in frame.columns:
        changed = frame["membership_changed"].fillna(False).astype(bool)
        summary["membership_change_days"] = {
            "days": int(changed.sum()),
            "share_of_days": float(changed.mean()),
            "share_of_absolute_bridge": float(frame.loc[changed, "abs_bridge"].sum() / total)
            if total else np.nan,
            "mean_absolute_bridge_on_change_days": float(frame.loc[changed, "abs_bridge"].mean())
            if changed.any() else np.nan,
            "mean_absolute_bridge_otherwise": float(frame.loc[~changed, "abs_bridge"].mean())
            if (~changed).any() else np.nan}
    if "days_since_snapshot" in frame.columns:
        staleness = pd.to_numeric(frame["days_since_snapshot"], errors="coerce")
        buckets = pd.cut(staleness, [-0.5, 0.5, 5.5, 10.5, 20.5, np.inf],
                         labels=["0", "1-5", "6-10", "11-20", "21+"])
        grouped = frame.assign(bucket=buckets).groupby("bucket", observed=True)["abs_bridge"]
        summary["by_snapshot_staleness"] = {
            str(name): {"days": int(len(values)), "mean_absolute_bridge": float(values.mean())}
            for name, values in grouped}
    return summary


def reproduction_verdict(rebuilt, saved, *, absolute_tolerance=1e-10, relative_tolerance=1e-6):
    """May the rebuilt basket be decomposed, or is it too far from the saved one?

    The gate is `max(absolute, relative * |saved|)`. The relative arm exists because Qlib
    stores prices as float32: a basket rebuilt from a fresh load agrees with the saved one
    to float32 precision (~1e-7 relative), not to float64. Judging that by an absolute
    1e-10 would reject a faithful rebuild. What makes the looser arm safe is scale — the
    per-security contributions being ranked are basis points (~1e-4), four to five orders
    of magnitude above this noise, so it cannot reorder them.
    """
    error = float(rebuilt) - float(saved)
    tolerance = max(float(absolute_tolerance), float(relative_tolerance) * abs(float(saved)))
    return {"rebuilt": float(rebuilt), "saved": float(saved), "error": error,
            "relative_error": error / abs(float(saved)) if saved else float("nan"),
            "tolerance": tolerance, "reproduced": abs(error) <= tolerance}


def basket_contributions(weights, returns):
    """Per-security `wb_i * r_i`, reproducing `attribute_d11`'s basket return.

    A constituent with no return that day is *dropped*, exactly as the attribution drops
    it — not treated as a zero return. Its weight is reported instead, because whatever
    the index earned on it lands in the bridge and cannot be attributed to any security.
    """
    weights = pd.Series(weights, dtype=float)
    returns = pd.Series(returns, dtype=float).reindex(weights.index)
    observable = returns.notna()
    contribution = (weights * returns).where(observable)
    return pd.DataFrame({"benchmark_weight": weights, "return": returns,
                         "contribution": contribution, "observable": observable}), {
        "basket_return": float(contribution[observable].sum()),
        "observable_weight": float(weights[observable].sum()),
        "unobservable_weight": float(weights[~observable].sum()),
        "unobservable_names": sorted(weights.index[~observable])}

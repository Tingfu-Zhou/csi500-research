from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.date_utils import parse_date_series


def to_qlib_symbol(ts_code: str) -> str:
    code, exchange = str(ts_code).upper().split(".")
    exchange_map = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}
    if exchange not in exchange_map:
        raise ValueError(f"未知交易所代码：{ts_code}")
    return f"{exchange_map[exchange]}{code}"


def to_tushare_symbol(instrument: str) -> str:
    instrument = str(instrument).upper()
    prefix, code = instrument[:2], instrument[2:]
    exchange_map = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}
    if prefix not in exchange_map or len(code) != 6:
        raise ValueError(f"未知 Qlib 股票代码：{instrument}")
    return f"{code}.{exchange_map[prefix]}"


def apply_membership_replacements(
    intervals: pd.DataFrame, replacements: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply explicit, auditable corrections to inclusive membership intervals."""
    result = intervals.copy()
    result["start_date"] = pd.to_datetime(result["start_date"]).dt.normalize()
    result["end_date"] = pd.to_datetime(result["end_date"]).dt.normalize()
    audit_rows: list[dict] = []

    for number, replacement in enumerate(replacements, start=1):
        start = pd.Timestamp(replacement["effective_start"]).normalize()
        end = pd.Timestamp(replacement["effective_end"]).normalize()
        outgoing = str(replacement["remove_instrument"]).upper()
        incoming = str(replacement["add_instrument"]).upper()
        if start > end or outgoing == incoming:
            raise ValueError(f"第 {number} 条成分修正配置无效")

        incoming_overlap = result.loc[
            result["instrument"].eq(incoming)
            & result["start_date"].le(end)
            & result["end_date"].ge(start)
        ]
        if not incoming_overlap.empty:
            raise ValueError(
                f"第 {number} 条成分修正与 {incoming} 的原始区间重叠"
            )

        revised_rows: list[dict] = []
        removed_overlap = False
        for row in result.to_dict("records"):
            row_start = pd.Timestamp(row["start_date"])
            row_end = pd.Timestamp(row["end_date"])
            overlaps = (
                row["instrument"] == outgoing
                and row_start <= end
                and row_end >= start
            )
            if not overlaps:
                revised_rows.append(row)
                continue
            removed_overlap = True
            if row_start < start:
                left = row.copy()
                left["end_date"] = start - pd.Timedelta(days=1)
                revised_rows.append(left)
            if row_end > end:
                right = row.copy()
                right["start_date"] = end + pd.Timedelta(days=1)
                revised_rows.append(right)

        if not removed_overlap:
            raise ValueError(
                f"第 {number} 条成分修正未命中 {outgoing} 的原始区间"
            )
        revised_rows.append(
            {"instrument": incoming, "start_date": start, "end_date": end}
        )
        result = pd.DataFrame(revised_rows, columns=result.columns)
        audit_rows.append(
            {
                "correction_id": number,
                "effective_start": start,
                "effective_end": end,
                "remove_instrument": outgoing,
                "add_instrument": incoming,
                "anchor_snapshot_date": pd.Timestamp(
                    replacement["anchor_snapshot_date"]
                ).normalize(),
                "reason": replacement.get("reason", ""),
                "evidence": replacement.get("evidence", ""),
            }
        )

    result = result.sort_values(["instrument", "start_date", "end_date"]).reset_index(
        drop=True
    )
    if result["start_date"].gt(result["end_date"]).any():
        raise ValueError("成分修正后产生了无效区间")
    return result, pd.DataFrame(audit_rows)


def add_backcast_snapshots(
    snapshots: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    replacements: list[dict],
) -> pd.DataFrame:
    """Backcast an observed later snapshot to a special-adjustment effective date."""
    result = normalize_snapshots(snapshots)
    if "snapshot_origin" not in result:
        result["snapshot_origin"] = "tushare_index_weight"

    close = adjusted_close.copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close.sort_index()

    synthetic: list[pd.DataFrame] = []
    for number, replacement in enumerate(replacements, start=1):
        effective = pd.Timestamp(replacement["effective_start"]).normalize()
        anchor = pd.Timestamp(replacement["anchor_snapshot_date"]).normalize()
        outgoing = str(replacement["remove_instrument"]).upper()
        incoming = str(replacement["add_instrument"]).upper()
        if anchor <= effective:
            raise ValueError(f"第 {number} 条修正的锚定快照必须晚于生效日")
        if result["trade_date"].eq(effective).any():
            raise ValueError(f"{effective.date()} 已有真实权重快照，无需反推")

        anchor_rows = result.loc[result["trade_date"].eq(anchor)].copy()
        if anchor_rows.empty:
            raise ValueError(f"缺少 {anchor.date()} 锚定权重快照")
        anchor_weights = anchor_rows.set_index("instrument")["weight"]
        if incoming not in anchor_weights.index or outgoing in anchor_weights.index:
            raise ValueError(f"{anchor.date()} 锚定快照与成分修正方向不一致")

        window = close.loc[close.index <= anchor, anchor_weights.index].where(
            lambda frame: frame.gt(0)
        )
        if window.loc[window.index >= effective].empty:
            raise ValueError(f"{effective.date()} 至 {anchor.date()} 没有可用行情")

        gross_returns = {}
        for instrument in anchor_weights.index:
            series = window[instrument].dropna()
            end_observed = series.loc[series.index <= anchor]
            if end_observed.empty:
                gross_returns[instrument] = 1.0
                continue
            if instrument == incoming:
                start_observed = series.loc[series.index >= effective]
                if start_observed.empty:
                    gross_returns[instrument] = 1.0
                    continue
                base = start_observed.iloc[0]
            else:
                start_observed = series.loc[series.index <= effective]
                if start_observed.empty:
                    gross_returns[instrument] = 1.0
                    continue
                base = start_observed.iloc[-1]
            gross_returns[instrument] = float(end_observed.iloc[-1] / base)
        gross = pd.Series(gross_returns).replace([np.inf, -np.inf], np.nan)
        if gross.isna().any() or gross.le(0).any():
            raise ValueError(f"第 {number} 条修正无法计算反推收益")

        effective_weights = anchor_weights / gross
        effective_weights = effective_weights / effective_weights.sum()
        synthetic.append(
            pd.DataFrame(
                {
                    "trade_date": effective,
                    "instrument": effective_weights.index,
                    "weight": effective_weights.values,
                    "snapshot_origin": "backcast_membership_correction",
                }
            )
        )

    if synthetic:
        result = pd.concat([result, *synthetic], ignore_index=True)
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def expand_membership(
    intervals: pd.DataFrame, calendar: Iterable[pd.Timestamp]
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values()
    pieces: list[pd.DataFrame] = []
    for row in intervals.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        active = dates[(dates >= start) & (dates <= end)]
        if len(active):
            pieces.append(
                pd.DataFrame({"trade_date": active, "instrument": row.instrument})
            )
    if not pieces:
        return pd.DataFrame(columns=["trade_date", "instrument"])
    result = pd.concat(pieces, ignore_index=True)
    duplicate_count = result.duplicated(["trade_date", "instrument"]).sum()
    if duplicate_count:
        raise ValueError(f"成分区间展开后存在 {duplicate_count} 条重复记录")
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def normalize_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    result = snapshots.copy()
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    if result["weight"].isna().any() or (result["weight"] < 0).any():
        raise ValueError("权重快照包含缺失、非数值或负权重")
    sums = result.groupby("trade_date")["weight"].sum()
    if sums.empty or (sums <= 0).any():
        raise ValueError("权重快照为空或权重和非正")
    if float(sums.median()) > 2.0:
        result["weight"] = result["weight"] / 100.0
    result["weight"] = result["weight"] / result.groupby("trade_date")[
        "weight"
    ].transform("sum")
    return result


def compare_snapshot_membership(
    membership: pd.DataFrame, snapshots: pd.DataFrame
) -> pd.DataFrame:
    member_sets = membership.groupby("trade_date")["instrument"].agg(set)
    snapshot_sets = snapshots.groupby("trade_date")["instrument"].agg(set)
    rows = []
    for date, snapshot_set in snapshot_sets.items():
        if date not in member_sets.index:
            rows.append(
                {
                    "trade_date": date,
                    "missing_in_snapshot": 0,
                    "extra_in_snapshot": len(snapshot_set),
                    "status": "date_not_in_membership_calendar",
                }
            )
            continue
        member_set = member_sets.loc[date]
        rows.append(
            {
                "trade_date": date,
                "missing_in_snapshot": len(member_set - snapshot_set),
                "extra_in_snapshot": len(snapshot_set - member_set),
                "status": "ok" if member_set == snapshot_set else "mismatch",
            }
        )
    return pd.DataFrame(rows)


def build_daily_weights(
    calendar: Iterable[pd.Timestamp],
    membership: pd.DataFrame,
    snapshots: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    output_start,
    output_end,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values()
    snapshots = normalize_snapshots(snapshots)
    snapshot_map = {
        date: group.set_index("instrument")["weight"].astype(float)
        for date, group in snapshots.groupby("trade_date")
    }
    if "snapshot_origin" in snapshots:
        origin_counts = snapshots.groupby("trade_date")["snapshot_origin"].nunique()
        if origin_counts.gt(1).any():
            raise ValueError("同一权重快照日期存在多个来源")
        snapshot_origin_map = snapshots.groupby("trade_date")["snapshot_origin"].first()
    else:
        snapshot_origin_map = pd.Series("unspecified", index=snapshot_map.keys())
    member_map = {
        date: set(group["instrument"])
        for date, group in membership.groupby("trade_date")
    }
    close = adjusted_close.copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close.sort_index()

    current: pd.Series | None = None
    previous_close: pd.Series | None = None
    active_snapshot_date: pd.Timestamp | None = None
    active_snapshot_origin: str | None = None
    rows: list[pd.DataFrame] = []

    for date in dates:
        members = member_map.get(date, set())
        if date in snapshot_map:
            current = snapshot_map[date].copy()
            active_snapshot_date = pd.Timestamp(date)
            active_snapshot_origin = str(snapshot_origin_map.loc[date])
            if set(current.index) != members:
                raise ValueError(f"{date.date()} 快照名单与 PIT 成分名单不一致")
        elif current is not None:
            if set(current.index) != members:
                raise ValueError(f"{date.date()} 成分名单变化但当天没有权重快照")
            today_close = close.loc[date].reindex(current.index)
            if previous_close is None:
                gross_return = pd.Series(1.0, index=current.index)
            else:
                prev = previous_close.reindex(current.index)
                gross_return = (today_close / prev).replace([np.inf, -np.inf], np.nan)
                gross_return = gross_return.fillna(1.0).clip(lower=0.0)
            current = current * gross_return
            if not np.isfinite(current).all() or current.sum() <= 0:
                raise ValueError(f"{date.date()} 漂移后权重无效")
            current = current / current.sum()

        if current is not None:
            today_close = close.loc[date].reindex(current.index)
            if previous_close is None:
                previous_close = today_close
            else:
                previous_close = today_close.combine_first(previous_close.reindex(current.index))

            if pd.Timestamp(output_start) <= date <= pd.Timestamp(output_end):
                rows.append(
                    pd.DataFrame(
                        {
                            "trade_date": date,
                            "instrument": current.index,
                            "weight": current.values,
                            "snapshot_date": active_snapshot_date,
                            "snapshot_origin": active_snapshot_origin,
                        }
                    )
                )

    if not rows:
        raise ValueError("没有生成任何日度基准权重")
    result = pd.concat(rows, ignore_index=True)
    expected_dates = dates[(dates >= pd.Timestamp(output_start)) & (dates <= pd.Timestamp(output_end))]
    missing_dates = expected_dates.difference(result["trade_date"].unique())
    if len(missing_dates):
        raise ValueError(f"日度权重缺少 {len(missing_dates)} 个交易日；需要更早的权重快照")
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def relative_error(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = right.abs().replace(0, np.nan)
    return (left - right).abs() / denominator


def classify_limit_quality(limits: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Separate valid all-zero placeholders from genuinely invalid limit rows."""
    required = ["pre_close", "up_limit", "down_limit"]
    missing = [column for column in required if column not in limits]
    if missing:
        raise ValueError(f"stk_limit 缺少字段：{missing}")
    numeric = limits[required].apply(pd.to_numeric, errors="coerce")
    zero_placeholder = numeric.eq(0).all(axis=1)
    anomaly = ~zero_placeholder & (
        numeric["up_limit"].le(0)
        | numeric["down_limit"].le(0)
        | numeric["up_limit"].lt(numeric["down_limit"])
    )
    return zero_placeholder, anomaly


def make_crosscheck(
    qlib_frame: pd.DataFrame,
    tushare_daily: pd.DataFrame,
    tushare_factor: pd.DataFrame,
) -> pd.DataFrame:
    q = qlib_frame.copy()
    q["trade_date"] = parse_date_series(q["trade_date"])
    q["ts_code"] = q["instrument"].map(to_tushare_symbol)
    q["qlib_raw_close"] = q["close"] / q["factor"]

    d = tushare_daily.copy()
    d["trade_date"] = parse_date_series(d["trade_date"])
    d = d.rename(columns={"close": "tushare_raw_close"})
    f = tushare_factor.copy()
    f["trade_date"] = parse_date_series(f["trade_date"])
    f = f.rename(columns={"adj_factor": "tushare_factor"})

    merged = q.merge(
        d[["ts_code", "trade_date", "tushare_raw_close"]],
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    ).merge(
        f[["ts_code", "trade_date", "tushare_factor"]],
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    merged["raw_close_relative_error"] = relative_error(
        merged["qlib_raw_close"], merged["tushare_raw_close"]
    )
    merged = merged.sort_values(["instrument", "trade_date"])
    merged["qlib_factor_ratio"] = merged["factor"] / merged.groupby("instrument")[
        "factor"
    ].transform("first")
    merged["tushare_factor_ratio"] = merged["tushare_factor"] / merged.groupby(
        "instrument"
    )["tushare_factor"].transform("first")
    merged["factor_ratio_relative_error"] = relative_error(
        merged["qlib_factor_ratio"], merged["tushare_factor_ratio"]
    )
    return merged.reset_index(drop=True)


def trade_day_age(
    trade_dates: pd.Series, list_dates: pd.Series, full_calendar: Iterable[pd.Timestamp]
) -> np.ndarray:
    calendar = np.asarray(pd.DatetimeIndex(pd.to_datetime(list(full_calendar))).values)
    dates = parse_date_series(trade_dates).values
    listed = parse_date_series(list_dates).values
    date_positions = np.searchsorted(calendar, dates, side="right") - 1
    list_positions = np.searchsorted(calendar, listed, side="left")
    return date_positions - list_positions + 1


def is_close(
    left: pd.Series, right: pd.Series, rtol: float, atol: float
) -> pd.Series:
    return pd.Series(
        np.isclose(left.astype(float), right.astype(float), rtol=rtol, atol=atol),
        index=left.index,
    )

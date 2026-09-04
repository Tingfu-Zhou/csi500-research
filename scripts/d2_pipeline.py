from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d2_core import (  # noqa: E402
    add_backcast_snapshots,
    apply_membership_replacements,
    build_daily_weights,
    classify_limit_quality,
    compare_snapshot_membership,
    expand_membership,
    is_close,
    make_crosscheck,
    normalize_snapshots,
    to_qlib_symbol,
    trade_day_age,
)
from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def request(method, *args, **kwargs) -> pd.DataFrame:
    for attempt in range(6):
        try:
            time.sleep(0.4)
            return method(*args, **kwargs)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = min(60, 5 * 2**attempt)
            print(f"请求失败：{exc}；{wait} 秒后重试", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_intervals(config: dict, end_date: pd.Timestamp) -> pd.DataFrame:
    path = Path(config["data"]["qlib_provider_uri"]) / "instruments/csi500.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到中证500成分区间文件：{path}")
    intervals = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["instrument", "start_date", "end_date"],
    )
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="raise")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="raise")
    intervals = intervals.loc[intervals["start_date"] <= end_date].copy()
    intervals["end_date"] = intervals["end_date"].clip(upper=end_date)
    duplicate_count = intervals.duplicated().sum()
    if duplicate_count:
        raise ValueError(f"csi500.txt 存在 {duplicate_count} 条完全重复区间")
    return intervals


def qlib_features(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date,
    end_date,
    amount_multiplier: float,
) -> pd.DataFrame:
    fields = ["$open", "$high", "$low", "$close", "$vwap", "$volume", "$factor", "$amount"]
    raw = access.features(instruments, fields, start_date, end_date)
    frame = raw.reset_index().rename(columns={"datetime": "trade_date"})
    frame.columns = [str(column).removeprefix("$") for column in frame.columns]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    numeric = ["open", "high", "low", "close", "vwap", "volume", "factor", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    for column in ["open", "high", "low", "close", "vwap"]:
        frame[f"raw_{column}"] = frame[column] / frame["factor"]
    frame["amount_cny"] = frame["amount"] * amount_multiplier
    return frame.sort_values(["instrument", "trade_date"]).reset_index(drop=True)


def choose_sample(
    membership: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    stock_count: int,
    day_count: int,
    seed: int,
) -> tuple[list[str], list[pd.Timestamp]]:
    rng = np.random.default_rng(seed)
    available_pairs = prices.loc[
        prices["close"].notna() & prices["factor"].gt(0),
        ["trade_date", "instrument"],
    ].drop_duplicates()
    eligible = membership.merge(
        available_pairs, on=["trade_date", "instrument"], how="inner"
    )
    pair_sets = eligible.groupby("trade_date")["instrument"].agg(set)
    candidate_dates = calendar[calendar.isin(pair_sets.index)]
    if len(candidate_dates) < day_count:
        raise ValueError("可用于对账的交易日不足 20 天")

    for _ in range(500):
        dates = sorted(pd.to_datetime(rng.choice(candidate_dates, size=day_count, replace=False)))
        common = set.intersection(*(pair_sets.loc[date] for date in dates))
        if len(common) >= stock_count:
            stocks = sorted(rng.choice(sorted(common), size=stock_count, replace=False).tolist())
            return stocks, dates
    raise ValueError("无法找到在 20 个抽样日均有行情的 50 只成分股")


def fetch_crosscheck_source(
    config: dict, dates: list[pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = ROOT / config["data"]["tushare_raw_dir"] / "qc_sample"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for date in dates:
        key = date.strftime("%Y%m%d")
        if not (cache_dir / f"daily_{key}.parquet").exists() or not (
            cache_dir / f"adj_factor_{key}.parquet"
        ).exists():
            missing.append(date)
    pro = None
    if missing:
        token = os.environ.get("QLIB_TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("对账缓存尚未建立，需设置 QLIB_TUSHARE_TOKEN")
        pro = ts.pro_api(token)

    daily_frames = []
    factor_frames = []
    for index, date in enumerate(dates, start=1):
        key = date.strftime("%Y%m%d")
        daily_path = cache_dir / f"daily_{key}.parquet"
        factor_path = cache_dir / f"adj_factor_{key}.parquet"
        if not daily_path.exists():
            daily = request(
                pro.daily,
                trade_date=key,
                fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            )
            atomic_parquet(daily, daily_path)
        if not factor_path.exists():
            factor = request(
                pro.adj_factor,
                trade_date=key,
                fields="ts_code,trade_date,adj_factor",
            )
            atomic_parquet(factor, factor_path)
        daily_frames.append(pd.read_parquet(daily_path))
        factor_frames.append(pd.read_parquet(factor_path))
        print(f"对账源数据：{index}/{len(dates)} 个交易日", flush=True)
    return (
        pd.concat(daily_frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"]),
        pd.concat(factor_frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"]),
    )


def load_yearly(
    access: ResearchDataAccess,
    config: dict,
    prefix: str,
    date_col: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    frames = []
    for year in range(start_year, end_year + 1):
        relative = raw_dir / f"{prefix}_{year}.parquet"
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 {relative}；请先运行 python scripts/pull_tushare_d2.py"
            )
        frames.append(
            access.read_parquet(
                relative,
                date_col=date_col,
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result[date_col] = parse_date_series(result[date_col])
    return result


def make_st_pairs(
    namechange: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    design_start: pd.Timestamp,
    design_end: pd.Timestamp,
) -> pd.DataFrame:
    changes = namechange.loc[
        namechange["name"].astype(str).str.contains("ST", case=False, na=False)
    ].copy()
    changes["start_date"] = parse_date_series(changes["start_date"], errors="coerce")
    changes["end_date"] = parse_date_series(
        changes["end_date"], errors="coerce"
    ).fillna(design_end)
    changes["end_date"] = changes["end_date"].clip(upper=design_end)
    changes = changes.loc[
        changes["start_date"].le(design_end) & changes["end_date"].ge(design_start)
    ]
    intervals = changes.assign(instrument=changes["ts_code"].map(to_qlib_symbol))[
        ["instrument", "start_date", "end_date"]
    ]
    return expand_membership(intervals, calendar).assign(is_st=True)


def write_report(summary: dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "d2_qc_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    status = "通过" if summary["acceptance"]["passed"] else "未通过"
    lines = [
        "# D2 数据体检报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 核心结果",
        "",
        f"- 对账记录：{summary['crosscheck']['rows']:,}（目标 1,000）",
        f"- 收盘价最大相对误差：{summary['crosscheck']['max_raw_close_relative_error']:.6%}",
        f"- 复权因子相对变化最大误差：{summary['crosscheck']['max_factor_ratio_relative_error']:.6%}",
        f"- 日度权重交易日：{summary['weights']['trade_days']:,}",
        f"- 权重和最大误差：{summary['weights']['max_weight_sum_error']:.3e}",
        f"- 快照名单不一致日：{summary['weights']['snapshot_mismatch_days']}",
        f"- 成分数范围：{summary['weights']['min_constituents']}–{summary['weights']['max_constituents']}",
        f"- 已登记历史成分修正：{summary['weights']['membership_corrections']}",
        f"- 反推权重快照日：{summary['weights']['synthetic_snapshot_days']}",
        "",
        "## 原始数据检查",
        "",
    ]
    for name, item in summary["raw_checks"].items():
        placeholder_text = ""
        if "zero_limit_placeholders" in item:
            placeholder_text = (
                f"，合法全零占位 {item['zero_limit_placeholders']:,}"
            )
        lines.append(
            f"- {name}: {item['rows']:,} 行，重复 {item['duplicates']:,}，"
            f"缺失交易日 {item['missing_trade_days']:,}，异常 {item['anomalies']:,}"
            f"{placeholder_text}"
        )
    lines.append(f"- Qlib 行情异常：{summary['qlib_anomalies']:,}")
    lines.extend(
        [
            "",
            "## 可交易性标记",
            "",
            f"- 总记录：{summary['tradability']['rows']:,}",
            f"- 可买比例：{summary['tradability']['can_buy_rate']:.2%}",
            f"- 可卖比例：{summary['tradability']['can_sell_rate']:.2%}",
            f"- 停牌记录：{summary['tradability']['suspended_rows']:,}",
            f"- ST / *ST 记录：{summary['tradability']['st_rows']:,}",
            f"- 上市不足 120 交易日：{summary['tradability']['young_rows']:,}",
            f"- 一字涨停：{summary['tradability']['one_price_up_rows']:,}",
            f"- 一字跌停：{summary['tradability']['one_price_down_rows']:,}",
            "",
            "## 口径",
            "",
            "- Qlib 未复权价按 `$close / $factor` 还原。",
            "- Qlib factor 与 Tushare adj_factor 的绝对尺度不同，只比较同一股票内相对变化。",
            "- 月内基准权重用 Qlib 复权收盘价收益自然漂移，停牌/缺价日按零收益处理。",
            "- 特殊成分调整若无生效日权重，则由下一份 Tushare 权重快照按复权收益反推，并保留审计记录。",
            "- Qlib `$amount` 按千元转为人民币元，20 日均额门槛为 5,000 万元。",
        ]
    )
    (reports_dir / "D2数据体检报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    access = ResearchDataAccess(ROOT)
    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    start_year, end_year = design_start.year, design_end.year
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    processed_dir = ROOT / config["data"]["processed_design_dir"]
    reports_dir = ROOT / config["data"]["reports_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    calendar_frame = access.read_parquet(
        raw_dir / "trade_cal.parquet",
        date_col="cal_date",
        end_date=design_end,
    )
    calendar_frame["cal_date"] = parse_date_series(calendar_frame["cal_date"])
    full_open_calendar = pd.DatetimeIndex(
        calendar_frame.loc[calendar_frame["is_open"].astype(int).eq(1), "cal_date"]
    ).sort_values()
    design_calendar = full_open_calendar[
        (full_open_calendar >= design_start) & (full_open_calendar <= design_end)
    ]
    if design_calendar.empty:
        raise ValueError("设计窗口交易日历为空")

    intervals = load_intervals(config, design_end)
    replacements = config["universe"].get("membership_replacements", [])
    intervals, correction_audit = apply_membership_replacements(
        intervals, replacements
    )
    if not correction_audit.empty:
        atomic_parquet(
            correction_audit,
            reports_dir / "d2_membership_corrections.parquet",
        )
    weight_source = access.read_parquet(
        raw_dir / "index_weight_000905_SH.parquet",
        date_col="trade_date",
        end_date=design_end,
    )
    weight_source["trade_date"] = parse_date_series(weight_source["trade_date"])
    weight_source["instrument"] = weight_source["con_code"].map(to_qlib_symbol)
    weight_source["weight"] = pd.to_numeric(weight_source["weight"], errors="raise")
    snapshots = weight_source[["trade_date", "instrument", "weight"]].drop_duplicates()
    conflicting = snapshots.duplicated(["trade_date", "instrument"]).sum()
    if conflicting:
        raise ValueError(f"权重快照存在 {conflicting} 条重复股票记录")
    snapshots = normalize_snapshots(snapshots)
    snapshots["snapshot_origin"] = "tushare_index_weight"
    bootstrap = snapshots.loc[snapshots["trade_date"] <= design_start, "trade_date"].max()
    if pd.isna(bootstrap):
        raise ValueError("2014-01-01 之前没有启动权重快照；请先运行 D2 补数脚本")

    weight_calendar = full_open_calendar[
        (full_open_calendar >= bootstrap) & (full_open_calendar <= design_end)
    ]
    membership_all = expand_membership(intervals, weight_calendar)
    membership_design = membership_all.loc[
        membership_all["trade_date"].between(design_start, design_end)
    ].copy()
    snapshots = snapshots.loc[
        snapshots["trade_date"].between(weight_calendar.min(), design_end)
    ].copy()
    membership_report = compare_snapshot_membership(membership_all, snapshots)
    atomic_parquet(membership_report, reports_dir / "d2_snapshot_membership.parquet")

    lookback_start = min(bootstrap, design_start - pd.Timedelta(days=90))
    instruments = sorted(membership_all["instrument"].unique())
    prices = qlib_features(
        access,
        instruments,
        lookback_start,
        design_end,
        float(config["tradability"]["qlib_amount_multiplier_to_cny"]),
    )
    adjusted_close = prices.pivot(index="trade_date", columns="instrument", values="close")
    weight_snapshots = add_backcast_snapshots(
        snapshots,
        adjusted_close,
        replacements,
    )
    daily_weights = build_daily_weights(
        weight_calendar,
        membership_all,
        weight_snapshots,
        adjusted_close,
        design_start,
        design_end,
    )
    atomic_parquet(daily_weights, processed_dir / "benchmark_weights.parquet")

    stock_count = int(config["validation"]["sample_stock_count"])
    day_count = int(config["validation"]["sample_trade_day_count"])
    sample_stocks, sample_dates = choose_sample(
        membership_design,
        prices,
        design_calendar,
        stock_count,
        day_count,
        int(config["project"]["random_seed"]),
    )
    tushare_daily, tushare_factor = fetch_crosscheck_source(config, sample_dates)
    ts_common = set(tushare_daily["ts_code"]) & set(tushare_factor["ts_code"])
    sample_stocks = [
        stock
        for stock in sample_stocks
        if f"{stock[2:]}.{stock[:2]}" in ts_common
    ]
    if len(sample_stocks) != stock_count:
        raise ValueError("Tushare 对账源缺少部分抽样股票；请删除 qc_sample 后重试")
    qlib_sample = prices.loc[
        prices["instrument"].isin(sample_stocks)
        & prices["trade_date"].isin(sample_dates),
        ["instrument", "trade_date", "close", "factor"],
    ]
    crosscheck = make_crosscheck(qlib_sample, tushare_daily, tushare_factor)
    expected_rows = stock_count * day_count
    if len(crosscheck) != expected_rows:
        raise ValueError(f"对账合并后只有 {len(crosscheck)} 行，预期 {expected_rows} 行")
    atomic_parquet(crosscheck, reports_dir / "d2_crosscheck_detail.parquet")

    daily_basic = load_yearly(
        access, config, "daily_basic", "trade_date", start_year, end_year
    )
    limits = load_yearly(access, config, "stk_limit", "trade_date", start_year, end_year)
    suspensions = load_yearly(
        access, config, "suspend_d", "trade_date", start_year, end_year
    )
    raw_checks = {}
    for name, frame, keys in [
        ("daily_basic", daily_basic, ["trade_date", "ts_code"]),
        ("stk_limit", limits, ["trade_date", "ts_code"]),
        ("suspend_d", suspensions, ["trade_date", "ts_code", "suspend_timing", "suspend_type"]),
    ]:
        raw_checks[name] = {
            "rows": int(len(frame)),
            "duplicates": int(frame.duplicated(keys).sum()),
            "trade_days": int(frame["trade_date"].nunique()),
            "missing_trade_days": 0,
            "anomalies": 0,
        }

    expected_date_set = set(pd.DatetimeIndex(design_calendar))
    for name, frame in [("daily_basic", daily_basic), ("stk_limit", limits)]:
        raw_checks[name]["missing_trade_days"] = len(
            expected_date_set
            - set(pd.DatetimeIndex(parse_date_series(frame["trade_date"])).normalize())
        )
    daily_basic_columns = [
        column
        for column in ["close", "total_mv", "circ_mv", "turnover_rate"]
        if column in daily_basic
    ]
    daily_basic_numeric = daily_basic[daily_basic_columns].apply(pd.to_numeric, errors="coerce")
    raw_checks["daily_basic"]["anomalies"] = int(
        daily_basic_numeric.lt(0).any(axis=1).sum()
    )
    zero_limit_placeholders, invalid_limit_rows = classify_limit_quality(limits)
    raw_checks["stk_limit"]["zero_limit_placeholders"] = int(
        zero_limit_placeholders.sum()
    )
    raw_checks["stk_limit"]["anomalies"] = int(invalid_limit_rows.sum())

    amount_window = int(config["tradability"]["amount_lookback_trade_days"])
    prices["avg_amount_cny"] = prices.groupby("instrument", sort=False)["amount_cny"].transform(
        lambda series: series.fillna(0).rolling(amount_window, min_periods=amount_window).mean()
    )
    market_columns = [
        "trade_date",
        "instrument",
        "raw_high",
        "raw_low",
        "raw_close",
        "amount_cny",
        "avg_amount_cny",
    ]
    tradability = membership_design.merge(
        prices[market_columns], on=["trade_date", "instrument"], how="left", validate="one_to_one"
    )

    limits = limits.assign(instrument=limits["ts_code"].map(to_qlib_symbol))
    limits = limits.drop_duplicates(["trade_date", "instrument"])
    tradability = tradability.merge(
        limits[["trade_date", "instrument", "up_limit", "down_limit"]],
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    suspended_pairs = (
        suspensions.loc[suspensions["suspend_type"].eq("S")]
        .assign(instrument=lambda frame: frame["ts_code"].map(to_qlib_symbol))
        [["trade_date", "instrument"]]
        .drop_duplicates()
        .assign(suspend_record=True)
    )
    tradability = tradability.merge(
        suspended_pairs,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )

    stock_basic = access.read_parquet(
        raw_dir / "stock_basic.parquet",
        date_col="list_date",
        end_date=design_end,
    )
    stock_basic["list_date"] = parse_date_series(
        stock_basic["list_date"], errors="coerce"
    )
    listings = (
        stock_basic.assign(instrument=stock_basic["ts_code"].map(to_qlib_symbol))
        .groupby("instrument", as_index=False)["list_date"]
        .min()
    )
    tradability = tradability.merge(listings, on="instrument", how="left", validate="many_to_one")
    if tradability["list_date"].isna().any():
        raise ValueError("部分中证500股票在 stock_basic 中没有上市日期")
    tradability["listing_trade_days"] = trade_day_age(
        tradability["trade_date"], tradability["list_date"], full_open_calendar
    )

    namechange = access.read_parquet(
        raw_dir / "namechange.parquet",
        date_col="start_date",
        end_date=design_end,
    )
    st_pairs = make_st_pairs(namechange, design_calendar, design_start, design_end)
    tradability = tradability.merge(
        st_pairs[["trade_date", "instrument", "is_st"]].drop_duplicates(),
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    tradability["is_st"] = tradability["is_st"].eq(True)
    tradability["is_suspended"] = (
        tradability["suspend_record"].eq(True)
        | tradability["raw_close"].isna()
    )
    price_rtol = float(config["tradability"]["one_price_relative_tolerance"])
    price_atol = float(config["tradability"]["one_price_absolute_tolerance_cny"])
    one_price = is_close(
        tradability["raw_high"], tradability["raw_low"], price_rtol, price_atol
    )
    tradability["is_one_price_limit_up"] = (
        one_price
        & is_close(
            tradability["raw_close"], tradability["up_limit"], price_rtol, price_atol
        )
    ).fillna(False)
    tradability["is_one_price_limit_down"] = (
        one_price
        & is_close(
            tradability["raw_close"], tradability["down_limit"], price_rtol, price_atol
        )
    ).fillna(False)
    minimum_age = int(config["tradability"]["listing_min_trade_days"])
    tradability["is_listed_under_120d"] = tradability["listing_trade_days"] < minimum_age
    minimum_amount = float(config["tradability"]["min_avg_daily_amount_cny"])
    tradability["is_below_amount_floor"] = (
        tradability["avg_amount_cny"].isna() | tradability["avg_amount_cny"].lt(minimum_amount)
    )
    tradability["is_eligible"] = ~(
        tradability["is_st"]
        | tradability["is_listed_under_120d"]
        | tradability["is_below_amount_floor"]
    )
    tradability["can_buy"] = (
        tradability["is_eligible"]
        & ~tradability["is_suspended"]
        & ~tradability["is_one_price_limit_up"]
    )
    tradability["can_sell"] = (
        ~tradability["is_suspended"] & ~tradability["is_one_price_limit_down"]
    )
    tradability = tradability.drop(columns=["suspend_record"])
    atomic_parquet(
        tradability.sort_values(["trade_date", "instrument"]),
        processed_dir / "tradability_flags.parquet",
    )

    tolerance = float(config["validation"]["relative_tolerance"])
    weight_sums = daily_weights.groupby("trade_date")["weight"].sum()
    counts = daily_weights.groupby("trade_date")["instrument"].nunique()
    max_close_error = float(crosscheck["raw_close_relative_error"].max())
    max_factor_error = float(crosscheck["factor_ratio_relative_error"].max())
    mismatch_days = int(membership_report["status"].ne("ok").sum())
    max_weight_error = float((weight_sums - 1.0).abs().max())
    duplicate_total = sum(item["duplicates"] for item in raw_checks.values())
    missing_trade_days = sum(item["missing_trade_days"] for item in raw_checks.values())
    raw_anomalies = sum(item["anomalies"] for item in raw_checks.values())
    qlib_anomalies = int(
        (
            (prices["factor"].notna() & prices["factor"].le(0))
            | (
                prices["raw_high"].notna()
                & prices["raw_low"].notna()
                & prices["raw_high"].lt(prices["raw_low"])
            )
            | (prices["amount_cny"].notna() & prices["amount_cny"].lt(0))
        ).sum()
    )
    tradability_duplicates = int(
        tradability.duplicated(["trade_date", "instrument"]).sum()
    )
    passed = bool(
        len(crosscheck) == expected_rows
        and max_close_error <= tolerance
        and max_factor_error <= tolerance
        and mismatch_days == 0
        and max_weight_error <= float(config["benchmark"]["weight_sum_tolerance"])
        and duplicate_total == 0
        and missing_trade_days == 0
        and raw_anomalies == 0
        and qlib_anomalies == 0
        and tradability_duplicates == 0
        and counts.eq(500).all()
        and weight_sums.index.min() == design_calendar.min()
        and weight_sums.index.max() == design_calendar.max()
    )
    summary = {
        "crosscheck": {
            "rows": int(len(crosscheck)),
            "stocks": stock_count,
            "trade_days": day_count,
            "max_raw_close_relative_error": max_close_error,
            "max_factor_ratio_relative_error": max_factor_error,
            "tolerance": tolerance,
        },
        "weights": {
            "rows": int(len(daily_weights)),
            "trade_days": int(daily_weights["trade_date"].nunique()),
            "max_weight_sum_error": max_weight_error,
            "snapshot_mismatch_days": mismatch_days,
            "min_constituents": int(counts.min()),
            "max_constituents": int(counts.max()),
            "membership_corrections": int(len(correction_audit)),
            "synthetic_snapshot_days": int(
                weight_snapshots.loc[
                    weight_snapshots["snapshot_origin"].eq(
                        "backcast_membership_correction"
                    ),
                    "trade_date",
                ].nunique()
            ),
        },
        "raw_checks": raw_checks,
        "qlib_anomalies": qlib_anomalies,
        "tradability": {
            "rows": int(len(tradability)),
            "duplicates": tradability_duplicates,
            "can_buy_rate": float(tradability["can_buy"].mean()),
            "can_sell_rate": float(tradability["can_sell"].mean()),
            "suspended_rows": int(tradability["is_suspended"].sum()),
            "st_rows": int(tradability["is_st"].sum()),
            "young_rows": int(tradability["is_listed_under_120d"].sum()),
            "one_price_up_rows": int(tradability["is_one_price_limit_up"].sum()),
            "one_price_down_rows": int(tradability["is_one_price_limit_down"].sum()),
        },
        "acceptance": {"passed": passed},
    }
    write_report(summary, reports_dir)
    print(f"D2 数据体检：{'通过' if passed else '未通过'}")
    print(f"报告：{reports_dir / 'D2数据体检报告.md'}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

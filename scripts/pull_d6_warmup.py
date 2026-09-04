from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.date_utils import parse_date_series  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def tushare_client():
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN")
    return ts.pro_api(token) if token else ts.pro_api()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def merge_before_design(
    new: pd.DataFrame,
    path: Path,
    *,
    design_start: pd.Timestamp,
    keys: list[str],
) -> pd.DataFrame:
    fresh = new.copy()
    fresh_dates = parse_date_series(fresh["trade_date"])
    fresh = fresh.loc[fresh_dates.lt(design_start)].copy()
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([fresh, existing], ignore_index=True, sort=False)
    else:
        combined = fresh
    combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values(keys).reset_index(drop=True)


def existing_pre_design_dates(path: Path, design_start: pd.Timestamp) -> int:
    if not path.exists():
        return 0
    frame = pd.read_parquet(path, columns=["trade_date"])
    dates = parse_date_series(frame["trade_date"], errors="coerce")
    return int(dates.loc[dates.lt(design_start)].nunique())


def pull_benchmark_warmup(
    pro,
    path: Path,
    *,
    start_date: pd.Timestamp,
    design_start: pd.Timestamp,
    required_price_dates: int,
) -> None:
    existing = existing_pre_design_dates(path, design_start)
    if existing >= int(required_price_dates):
        print(f"H00905.CSI 预热已存在：{existing} 个日期，跳过拉取")
        return
    print("拉取 H00905.CSI 设计窗口前预热行情……", flush=True)
    frame = pro.index_daily(
        ts_code="H00905.CSI",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=(design_start - pd.Timedelta(days=1)).strftime("%Y%m%d"),
    )
    if frame is None or frame.empty:
        raise RuntimeError("Tushare 未返回 H00905.CSI 设计窗口前行情")
    if {"trade_date", "close"} - set(frame.columns):
        raise RuntimeError(f"H00905.CSI 返回字段异常：{list(frame.columns)}")
    combined = merge_before_design(
        frame,
        path,
        design_start=design_start,
        keys=["trade_date"],
    )
    atomic_parquet(combined, path)
    count = existing_pre_design_dates(path, design_start)
    if count < int(required_price_dates):
        raise RuntimeError(
            f"H00905.CSI 预热只有 {count} 个价格日期，至少需要 {required_price_dates} 个"
        )
    print(f"H00905.CSI 预热完成：{count} 个设计窗口前价格日期")


def pull_daily_basic_warmup(
    pro,
    raw_dir: Path,
    *,
    start_date: pd.Timestamp,
    design_start: pd.Timestamp,
    required_dates: int,
) -> None:
    year_paths = [
        raw_dir / f"daily_basic_{year}.parquet"
        for year in range(start_date.year, design_start.year)
    ]
    existing_dates = sum(
        existing_pre_design_dates(path, design_start) for path in year_paths
    )
    if existing_dates >= int(required_dates):
        print(f"daily_basic 预热已存在：{existing_dates} 个日期，跳过拉取")
        return

    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=(design_start - pd.Timedelta(days=1)).strftime("%Y%m%d"),
        is_open=1,
        fields="cal_date,is_open",
    )
    if calendar is None or calendar.empty:
        raise RuntimeError("Tushare 未返回 daily_basic 预热交易日历")
    dates = sorted(calendar["cal_date"].astype(str).unique())
    print(f"拉取 daily_basic 预热：{len(dates)} 个交易日……", flush=True)
    frames = []
    for number, date in enumerate(dates, start=1):
        frame = pro.daily_basic(
            trade_date=date,
            fields="ts_code,trade_date,turnover_rate,total_mv",
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"Tushare daily_basic({date}) 返回为空")
        frames.append(frame)
        print(f"daily_basic 预热进度：{number}/{len(dates)}", flush=True)
        time.sleep(0.35)
    fresh = pd.concat(frames, ignore_index=True, sort=False)
    fresh_dates = parse_date_series(fresh["trade_date"])
    for year, group in fresh.groupby(fresh_dates.dt.year):
        path = raw_dir / f"daily_basic_{int(year)}.parquet"
        combined = merge_before_design(
            group,
            path,
            design_start=design_start,
            keys=["trade_date", "ts_code"],
        )
        atomic_parquet(combined, path)
    available = sum(existing_pre_design_dates(path, design_start) for path in year_paths)
    if available < int(required_dates):
        raise RuntimeError(
            f"daily_basic 预热只有 {available} 个日期，至少需要 {required_dates} 个"
        )
    print(f"daily_basic 预热完成：{available} 个设计窗口前日期")


def main() -> None:
    config = load_config()
    d6 = config["d6"]
    design_start = pd.Timestamp(config["windows"]["design_start"])
    history_start = design_start - pd.Timedelta(
        days=int(d6["history_buffer_calendar_days"])
    )
    basic_start = design_start - pd.Timedelta(
        days=int(d6["daily_basic_warmup_calendar_days"])
    )
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    benchmark_path = raw_dir / "index_daily_H00905_CSI.parquet"
    pro = tushare_client()

    pull_benchmark_warmup(
        pro,
        benchmark_path,
        start_date=history_start,
        design_start=design_start,
        required_price_dates=int(d6["beta_window_trade_days"]) + 1,
    )
    pull_daily_basic_warmup(
        pro,
        raw_dir,
        start_date=basic_start,
        design_start=design_start,
        required_dates=int(d6["liquidity_window_trade_days"]),
    )
    print("D6 预热数据准备完成；只新增 2014-01-01 以前的数据。")
    print("请继续运行 python scripts/seal_oos.py verify。")


if __name__ == "__main__":
    main()

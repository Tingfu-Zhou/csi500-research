from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.date_utils import parse_date_series  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def request(method, delay: float, *args, **kwargs) -> pd.DataFrame:
    for attempt in range(6):
        try:
            time.sleep(delay)
            return method(*args, **kwargs)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = min(60, 5 * 2**attempt)
            print(f"请求失败：{exc}；{wait} 秒后重试", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def atomic_parquet(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.drop_duplicates()
    columns = [column for column in sort_columns if column in result.columns]
    if columns:
        result = result.sort_values(columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(path)


def ensure_long_trade_calendar(pro, config: dict, delay: float) -> pd.DataFrame:
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    path = raw_dir / "trade_cal.parquet"
    end_year = pd.Timestamp(config["data"]["end_date"]).year
    frames = []
    for year in range(1990, end_year + 1):
        frames.append(
            request(
                pro.trade_cal,
                delay,
                exchange="SSE",
                start_date=f"{year}0101",
                end_date=f"{year}1231",
                fields="exchange,cal_date,is_open,pretrade_date",
            )
        )
    calendar = pd.concat(frames, ignore_index=True)
    atomic_parquet(calendar, path, ["cal_date"])
    print(f"trade_cal.parquet 已扩展至 1990 年：{len(calendar):,} 行")
    return calendar


def ensure_bootstrap_weight(pro, config: dict, delay: float) -> None:
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    path = raw_dir / "index_weight_000905_SH.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少 D1 文件：{path}")
    existing = pd.read_parquet(path)
    start = pd.Timestamp(config["data"]["start_date"])
    months = int(config["benchmark"]["bootstrap_months"])
    periods = pd.period_range(start - pd.DateOffset(months=months), start - pd.Timedelta(days=1), freq="M")
    frames = [existing]
    for period in periods:
        frame = request(
            pro.index_weight,
            delay,
            index_code=config["benchmark"]["constituent_code"],
            start_date=period.start_time.strftime("%Y%m%d"),
            end_date=period.end_time.strftime("%Y%m%d"),
        )
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    atomic_parquet(combined, path, ["trade_date", "con_code"])
    first_date = parse_date_series(combined["trade_date"]).min().date()
    print(f"index_weight 已补启动快照；最早快照：{first_date}")


ENDPOINTS = {
    "daily_basic": {
        "method": "daily_basic",
        "fields": (
            "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
            "free_share,total_mv,circ_mv"
        ),
        "keys": ["trade_date", "ts_code"],
    },
    "stk_limit": {
        "method": "stk_limit",
        "fields": "trade_date,ts_code,pre_close,up_limit,down_limit",
        "keys": ["trade_date", "ts_code"],
    },
    "suspend_d": {
        "method": "suspend_d",
        "fields": "ts_code,trade_date,suspend_timing,suspend_type",
        "keys": ["trade_date", "ts_code", "suspend_timing"],
        "call_kwargs": {"suspend_type": "S"},
    },
}


def pull_month(pro, endpoint: str, dates: list[str], cache_path: Path, delay: float) -> pd.DataFrame:
    spec = ENDPOINTS[endpoint]
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    method = getattr(pro, spec["method"])
    frames = []
    for index, trade_date in enumerate(dates, start=1):
        frame = request(
            method,
            delay,
            trade_date=trade_date,
            fields=spec["fields"],
            **spec.get("call_kwargs", {}),
        )
        if not frame.empty:
            frames.append(frame)
        if index % 20 == 0 or index == len(dates):
            print(
                f"  {endpoint} {trade_date}: {index}/{len(dates)} 个交易日",
                flush=True,
            )
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=spec["fields"].split(","))
    atomic_parquet(result, cache_path, spec["keys"])
    return result


def parse_years(text: str | None, config: dict) -> list[int]:
    if text is None:
        start = pd.Timestamp(config["data"]["start_date"]).year
        end = pd.Timestamp(config["data"]["end_date"]).year
        return list(range(start, end + 1))
    years: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            left, right = (int(value) for value in part.split("-", maxsplit=1))
            years.update(range(left, right + 1))
        else:
            years.add(int(part))
    return sorted(years)


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取 D2 所需的逐日 Tushare 数据")
    parser.add_argument("--years", help="例如 2014-2018 或 2014,2016；默认全区间")
    parser.add_argument("--request-delay", type=float, default=0.4)
    parser.add_argument("--force", action="store_true", help="重建已存在的年度文件")
    args = parser.parse_args()

    config = load_config()
    marker = ROOT / config["data"]["window_marker"]
    if marker.exists():
        raise RuntimeError("窗口已经封存。补数必须在 seal_oos.py seal 之前完成。")
    token = os.environ.get("QLIB_TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置环境变量 QLIB_TUSHARE_TOKEN")
    pro = ts.pro_api(token)

    calendar = ensure_long_trade_calendar(pro, config, args.request_delay)
    ensure_bootstrap_weight(pro, config, args.request_delay)
    calendar["date"] = parse_date_series(calendar["cal_date"])
    open_calendar = calendar.loc[calendar["is_open"].astype(int).eq(1)].copy()

    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    sealed_dir = ROOT / config["data"]["tushare_sealed_dir"]
    design_end_year = pd.Timestamp(config["windows"]["design_end"]).year

    for year in parse_years(args.years, config):
        destination = raw_dir if year <= design_end_year else sealed_dir
        destination.mkdir(parents=True, exist_ok=True)
        year_dates = open_calendar.loc[
            open_calendar["date"].dt.year.eq(year), "date"
        ]
        if year_dates.empty:
            raise ValueError(f"交易日历中没有 {year} 年")

        for endpoint, spec in ENDPOINTS.items():
            annual_path = destination / f"{endpoint}_{year}.parquet"
            if annual_path.exists() and not args.force:
                print(f"跳过已有文件：{annual_path.relative_to(ROOT)}")
                continue
            frames = []
            for month, month_dates in year_dates.groupby(year_dates.dt.month):
                date_strings = month_dates.dt.strftime("%Y%m%d").tolist()
                cache_path = destination / ".d2_cache" / f"{endpoint}_{year}_{month:02d}.parquet"
                if args.force:
                    cache_path.unlink(missing_ok=True)
                frames.append(
                    pull_month(
                        pro,
                        endpoint,
                        date_strings,
                        cache_path,
                        args.request_delay,
                    )
                )
            annual = pd.concat(frames, ignore_index=True)
            atomic_parquet(annual, annual_path, spec["keys"])
            print(f"{annual_path.relative_to(ROOT)}：{len(annual):,} 行")

    print("D2 Tushare 补数完成。2019 年及以后年度文件已直接写入 sealed 目录。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断；月度缓存会保留，下次运行可续传。", file=sys.stderr)
        raise SystemExit(130)

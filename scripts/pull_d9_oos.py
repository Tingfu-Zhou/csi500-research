from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d9_pipeline import (  # noqa: E402
    atomic_json,
    load_config,
    verify_frozen_repository,
)
from scripts.pull_tushare_d2 import (  # noqa: E402
    ENDPOINTS,
    atomic_parquet,
    pull_month,
    request,
)
from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402


def validate_annual_frame(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    year: int,
    expected_dates: pd.DatetimeIndex,
) -> dict[str, object]:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"未知 Tushare 端点：{endpoint}")
    spec = ENDPOINTS[endpoint]
    required = set(str(spec["fields"]).split(","))
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{endpoint}_{year}.parquet 缺少字段：{sorted(missing_columns)}"
        )
    if frame.duplicated(list(spec["keys"])).any():
        raise ValueError(f"{endpoint}_{year}.parquet 存在重复主键")
    dates = parse_date_series(frame["trade_date"], errors="raise")
    if len(frame) and not dates.dt.year.eq(int(year)).all():
        raise ValueError(f"{endpoint}_{year}.parquet 含其他年份数据")
    actual_dates = pd.DatetimeIndex(dates.dropna().unique()).normalize().sort_values()
    expected = pd.DatetimeIndex(expected_dates).normalize().sort_values().unique()
    missing_dates = expected.difference(actual_dates)
    if endpoint in {"daily_basic", "stk_limit"} and len(missing_dates):
        preview = ", ".join(str(value.date()) for value in missing_dates[:10])
        raise ValueError(
            f"{endpoint}_{year}.parquet 缺少 {len(missing_dates)} 个交易日：{preview}"
        )
    return {
        "file": f"{endpoint}_{year}.parquet",
        "endpoint": endpoint,
        "year": int(year),
        "rows": len(frame),
        "trade_dates": len(actual_dates),
        "expected_trade_dates": len(expected),
        "missing_required_trade_dates": (
            len(missing_dates)
            if endpoint in {"daily_basic", "stk_limit"}
            else 0
        ),
        "passed": True,
    }


def load_oos_calendar(pro, config: dict, delay: float) -> pd.DatetimeIndex:
    start = pd.Timestamp(config["windows"]["oos_start"])
    end = pd.Timestamp(config["windows"]["oos_end"])
    frame = request(
        pro.trade_cal,
        delay,
        exchange="SSE",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    if frame.empty:
        raise ValueError("Tushare 未返回 D9 样本外交易日历")
    frame["cal_date"] = parse_date_series(frame["cal_date"], errors="raise")
    calendar = pd.DatetimeIndex(
        frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"]
    ).normalize().sort_values().unique()
    if calendar.empty or calendar.min() < start or calendar.max() > end:
        raise ValueError("D9 样本外交易日历范围无效")
    return calendar


def download_oos(config: dict, *, request_delay: float) -> dict[str, object]:
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("补齐 D9 OOS 原始文件时必须保持 design 模式，不得先解封")
    verify_frozen_repository(config)
    manifest_path = ROOT / config["d9"]["run_manifest"]
    summary_path = ROOT / config["data"]["reports_dir"] / "d9_summary.json"
    if manifest_path.exists() or summary_path.exists():
        raise RuntimeError("D9 已经启动或完成，禁止再补拉样本外原始文件")
    token = os.environ.get("QLIB_TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置环境变量 QLIB_TUSHARE_TOKEN")
    pro = ts.pro_api(token)
    calendar = load_oos_calendar(pro, config, float(request_delay))
    sealed = ROOT / config["data"]["tushare_sealed_dir"]
    sealed.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    for year in range(calendar.min().year, calendar.max().year + 1):
        year_dates = calendar[calendar.year == year]
        if year_dates.empty:
            raise ValueError(f"D9 交易日历缺少 {year} 年")
        for endpoint, spec in ENDPOINTS.items():
            annual_path = sealed / f"{endpoint}_{year}.parquet"
            if annual_path.exists():
                annual = pd.read_parquet(annual_path)
                audit = validate_annual_frame(
                    annual,
                    endpoint=endpoint,
                    year=year,
                    expected_dates=year_dates,
                )
                audit["source"] = "existing_validated"
                audit_rows.append(audit)
                print(f"已存在并通过校验：{annual_path.relative_to(ROOT)}", flush=True)
                continue
            frames: list[pd.DataFrame] = []
            for month in sorted(set(year_dates.month)):
                month_dates = year_dates[year_dates.month == month]
                cache_path = (
                    sealed
                    / "_d9_download_cache"
                    / f"{endpoint}_{year}_{month:02d}.parquet"
                )
                frames.append(
                    pull_month(
                        pro,
                        endpoint,
                        month_dates.strftime("%Y%m%d").tolist(),
                        cache_path,
                        float(request_delay),
                    )
                )
            annual = pd.concat(frames, ignore_index=True)
            audit = validate_annual_frame(
                annual,
                endpoint=endpoint,
                year=year,
                expected_dates=year_dates,
            )
            atomic_parquet(annual, annual_path, list(spec["keys"]))
            audit["source"] = "downloaded"
            audit_rows.append(audit)
            print(
                f"{annual_path.relative_to(ROOT)}：{len(annual):,} 行，校验通过",
                flush=True,
            )
    payload = {
        "purpose": "D9 OOS ingestion only; no model, return, or threshold calculation",
        "window_mode": access.mode,
        "oos_start": config["windows"]["oos_start"],
        "oos_end": config["windows"]["oos_end"],
        "formal_oos_run_started": False,
        "files": audit_rows,
        "passed": bool(len(audit_rows) == 24 and all(row["passed"] for row in audit_rows)),
    }
    if not payload["passed"]:
        raise RuntimeError("D9 OOS 原始文件未完整取得 24 个年度端点文件")
    atomic_json(payload, sealed / "d9_ingestion_manifest.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 design 模式下将 D9 所需 Tushare OOS 原始数据直接写入 sealed"
    )
    parser.add_argument("--request-delay", type=float, default=0.4)
    args = parser.parse_args()
    if not math.isfinite(float(args.request_delay)) or float(args.request_delay) < 0:
        raise ValueError("request-delay 必须是非负有限数")
    payload = download_oos(load_config(), request_delay=float(args.request_delay))
    print(
        f"D9 OOS 原始文件补齐：{len(payload['files'])}/24；"
        "仍为 design 模式，正式 OOS 尚未启动。"
    )
    print("下一步：python scripts/d9_pipeline.py --preflight")


if __name__ == "__main__":
    main()

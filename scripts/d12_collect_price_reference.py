"""Collect narrow direct-Tushare evidence for D12 P0-15/P0-17.

This script does not edit Qlib data or repair R0. It only stores direct Tushare
daily, adjustment-factor and share-count observations around the
300114.SZ -> 302132.SZ identifier change. The two cache filenames are shaped so
that a later immutable ``d12_day3.py`` run can read them independently of the
existing Qlib conversion. This is not a second-vendor validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CODES = ("300114.SZ", "302132.SZ")
START_DATE = "20250203"
END_DATE = "20250307"


def request(method, **kwargs):
    """Apply the same conservative retry policy as the existing Tushare pull."""
    for attempt in range(6):
        try:
            time.sleep(0.7)
            return method(**kwargs)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = min(60, 5 * 2**attempt)
            print(f"请求失败：{exc}，{wait} 秒后重试", flush=True)
            time.sleep(wait)


def normalize(frame, required, endpoint):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{endpoint} 缺少字段：{missing}")
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    if not result.trade_date.str.fullmatch(r"\d{8}").all():
        raise RuntimeError(f"{endpoint} 返回了无效 trade_date")
    result = result.drop_duplicates().sort_values(["ts_code", "trade_date"])
    if result.duplicated(["ts_code", "trade_date"]).any():
        raise RuntimeError(f"{endpoint} 存在冲突的证券-日期重复行")
    return result.reset_index(drop=True)


def build_summary(daily, factors, basic):
    """Build an auditable joined table without changing source observations."""
    daily_columns = [
        "ts_code", "trade_date", "open", "high", "low", "close",
        "pre_close", "change", "pct_chg", "vol", "amount",
    ]
    factor_columns = ["ts_code", "trade_date", "adj_factor"]
    basic_columns = [
        "ts_code", "trade_date", "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
    ]
    d = normalize(daily, daily_columns, "daily")[daily_columns]
    f = normalize(factors, factor_columns, "adj_factor")[factor_columns]
    b = normalize(basic, basic_columns, "daily_basic")[basic_columns]
    result = d.merge(f, on=["ts_code", "trade_date"], how="outer", validate="one_to_one")
    result = result.merge(b, on=["ts_code", "trade_date"], how="outer", validate="one_to_one")
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("ts_code", sort=False)
    result["adj_factor_ratio"] = result.adj_factor / grouped.adj_factor.shift()
    result["close_times_adj_factor"] = result.close * result.adj_factor
    result["adjusted_return_from_direct_fields"] = (
        result.close_times_adj_factor / grouped.close_times_adj_factor.shift() - 1
    )
    return result


def fetch_endpoint(pro, endpoint, fields):
    frames = []
    method = getattr(pro, endpoint)
    for code in CODES:
        print(f"Tushare {endpoint}: {code} {START_DATE}-{END_DATE}", flush=True)
        frame = request(
            method,
            ts_code=code,
            start_date=START_DATE,
            end_date=END_DATE,
            fields=fields,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"Tushare {endpoint} 未返回任何数据")
    return pd.concat(frames, ignore_index=True)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="price_ref_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise RuntimeError("请使用简单且全新的 run-id")

    token = os.environ.get("QLIB_TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置环境变量 QLIB_TUSHARE_TOKEN")
    import tushare as ts

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / config["data"]["tushare_raw_dir"] / "d12_evidence" / args.run_id
    report_dir = ROOT / "reports/d12/evidence" / args.run_id
    if raw_dir.exists() or report_dir.exists():
        raise FileExistsError("取证 run-id 不可复用；请保留旧目录并使用新 ID")
    raw_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    pro = ts.pro_api(token)
    daily = fetch_endpoint(
        pro,
        "daily",
        "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    )
    factors = fetch_endpoint(pro, "adj_factor", "ts_code,trade_date,adj_factor")
    basic = fetch_endpoint(
        pro,
        "daily_basic",
        "ts_code,trade_date,total_share,float_share,free_share,total_mv,circ_mv",
    )
    summary = build_summary(daily, factors, basic)

    # The day3 loader recognises these endpoint_YYYYMMDD names recursively.
    daily_path = raw_dir / "daily_20250224.parquet"
    factor_path = raw_dir / "adj_factor_20250224.parquet"
    basic_path = raw_dir / "daily_basic_20250224.parquet"
    normalize(daily, ["ts_code", "trade_date", "close"], "daily").to_parquet(daily_path, index=False)
    normalize(factors, ["ts_code", "trade_date", "adj_factor"], "adj_factor").to_parquet(factor_path, index=False)
    normalize(basic, ["ts_code", "trade_date", "total_share"], "daily_basic").to_parquet(basic_path, index=False)
    summary.to_csv(report_dir / "direct_tushare_reference.tsv", sep="\t", index=False)

    manifest = {
        "status": "completed",
        "diagnostic_only": True,
        "qlib_data_changed": False,
        "r0_repaired": False,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "codes": list(CODES),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "rows": {"daily": len(daily), "adj_factor": len(factors), "daily_basic": len(basic)},
        "raw_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (daily_path, factor_path, basic_path)
        },
        "next_command": "python -u scripts/d12_day3.py --run-id day3_v3 --replay",
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {report_dir}", flush=True)
    print("下一步使用新的 day3_v3 运行 ID；不要修改或删除 day3_v2。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

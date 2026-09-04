import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config/config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

START_DATE = CONFIG["data"]["start_date"].replace("-", "")
END_DATE = CONFIG["data"]["end_date"].replace("-", "")
RAW_DIR = ROOT / CONFIG["data"]["tushare_raw_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("QLIB_TUSHARE_TOKEN")
if not TOKEN:
    raise RuntimeError("未设置环境变量 QLIB_TUSHARE_TOKEN")

pro = ts.pro_api(TOKEN)


def request(method, *args, **kwargs):
    """限速并自动重试。"""
    for attempt in range(6):
        try:
            time.sleep(0.7)
            return method(*args, **kwargs)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = min(60, 5 * 2**attempt)
            print(f"请求失败：{exc}，{wait} 秒后重试")
            time.sleep(wait)


def save(df, filename, sort_columns=None):
    df = df.drop_duplicates()

    if sort_columns:
        columns = [c for c in sort_columns if c in df.columns]
        if columns:
            df = df.sort_values(columns)

    path = RAW_DIR / filename
    df.to_parquet(path, index=False)
    print(f"{filename}: {len(df):,} rows")


# 1. 交易日历
trade_cal = request(
    pro.trade_cal,
    exchange="SSE",
    start_date=START_DATE,
    end_date=END_DATE,
    fields="exchange,cal_date,is_open,pretrade_date",
)
save(trade_cal, "trade_cal.parquet", ["cal_date"])


# 2. 中证500全收益指数
index_daily = request(
    pro.index_daily,
    ts_code=CONFIG["benchmark"]["total_return_code"],
    start_date=START_DATE,
    end_date=END_DATE,
)
save(index_daily, "index_daily_H00905_CSI.parquet", ["trade_date"])


# 3. 中证500月度权重快照
weight_frames = []

for month in pd.period_range(
    pd.to_datetime(START_DATE),
    pd.to_datetime(END_DATE),
    freq="M",
):
    month_start = max(month.start_time, pd.to_datetime(START_DATE))
    month_end = min(month.end_time, pd.to_datetime(END_DATE))

    df = request(
        pro.index_weight,
        index_code=CONFIG["benchmark"]["constituent_code"],
        start_date=month_start.strftime("%Y%m%d"),
        end_date=month_end.strftime("%Y%m%d"),
    )

    if not df.empty:
        weight_frames.append(df)

index_weight = pd.concat(weight_frames, ignore_index=True)
save(
    index_weight,
    "index_weight_000905_SH.parquet",
    ["trade_date", "con_code"],
)


# 4. 股票基础信息，包括退市股票
stock_frames = []

for status in ["L", "D", "P"]:
    df = request(
        pro.stock_basic,
        exchange="",
        list_status=status,
        fields=(
            "ts_code,symbol,name,area,industry,market,"
            "list_date,delist_date,list_status"
        ),
    )
    stock_frames.append(df)

stock_basic = pd.concat(stock_frames, ignore_index=True)
save(stock_basic, "stock_basic.parquet", ["ts_code", "list_status"])


# 5. 股票名称和ST状态变更，分页获取
namechange_frames = []
offset = 0
limit = 1000

while True:
    df = request(
        pro.query,
        "namechange",
        fields=(
            "ts_code,name,start_date,end_date,"
            "ann_date,change_reason"
        ),
        limit=limit,
        offset=offset,
    )

    if df.empty:
        break

    namechange_frames.append(df)
    print(f"namechange：已取得 {offset + len(df):,} 行")

    if len(df) < limit:
        break

    offset += limit

namechange = pd.concat(namechange_frames, ignore_index=True)
save(namechange, "namechange.parquet", ["start_date", "ts_code"])

print("Tushare 核心数据拉取完成。")
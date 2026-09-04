from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d6_core import resolve_static_industry_mapping  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def tushare_client():
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN")
    return ts.pro_api(token) if token else ts.pro_api()


def fetch_catalog(pro) -> pd.DataFrame:
    raw = pro.index_classify(level="L1", src="SW2021")
    if raw is None or raw.empty:
        raise RuntimeError("Tushare index_classify 没有返回 SW2021 一级行业")
    code_column = "index_code" if "index_code" in raw.columns else "l1_code"
    required = {code_column, "industry_name"}
    if required - set(raw.columns):
        raise RuntimeError(f"Tushare 行业目录字段异常：{list(raw.columns)}")
    return raw.rename(columns={code_column: "l1_code"})[
        ["l1_code", "industry_name"]
    ].drop_duplicates()


def fetch_members(pro, catalog: pd.DataFrame) -> pd.DataFrame:
    frames = []
    total = len(catalog) * 2
    completed = 0
    for l1_code in catalog["l1_code"].sort_values():
        for is_new in ["Y", "N"]:
            frame = pro.index_member_all(l1_code=l1_code, is_new=is_new)
            completed += 1
            if frame is not None and not frame.empty:
                frame = frame.copy()
                if "l1_code" not in frame:
                    frame["l1_code"] = l1_code
                if "is_new" not in frame:
                    frame["is_new"] = is_new
                frames.append(frame)
            print(f"行业成员进度：{completed}/{total}", flush=True)
            time.sleep(0.12)
    if not frames:
        raise RuntimeError("Tushare index_member_all 没有返回任何行业成员")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    config = load_config()
    d6 = config["d6"]
    if d6.get("industry_version") != "SW2021":
        raise ValueError("D6 固定要求 industry_version=SW2021")

    print("读取 Tushare SW2021 一级行业目录……")
    pro = tushare_client()
    catalog_raw = fetch_catalog(pro)
    print(f"一级行业：{len(catalog_raw)} 个；读取当前与历史成员……")
    members_raw = fetch_members(pro, catalog_raw)
    mapping, catalog = resolve_static_industry_mapping(
        members_raw,
        catalog_raw,
        expected_industry_count=int(d6["expected_industry_count"]),
    )

    mapping_path = ROOT / d6["industry_mapping_file"]
    catalog_path = ROOT / d6["industry_catalog_file"]
    atomic_parquet(mapping, mapping_path)
    atomic_parquet(catalog, catalog_path)
    print(f"固定行业映射：{len(mapping):,} 只股票")
    print(f"行业目录：{catalog_path}")
    print(f"行业映射：{mapping_path}")
    print("说明：这是按计划固定的非 PIT 分类；文件不含历史生效日期。")


if __name__ == "__main__":
    main()

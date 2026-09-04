from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_access import ResearchDataAccess, WindowViolation  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402


CORE_SPECS = {
    "trade_cal.parquet": "cal_date",
    "index_daily_H00905_CSI.parquet": "trade_date",
    "index_weight_000905_SH.parquet": "trade_date",
    "namechange.parquet": "start_date",
    "stock_basic.parquet": "list_date",
}


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def design_copy(filename: str, frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    date_col = CORE_SPECS[filename]
    dates = parse_date_series(frame[date_col], errors="coerce")
    result = frame.loc[dates.le(cutoff)].copy()
    if filename == "namechange.parquet":
        end = parse_date_series(result["end_date"], errors="coerce")
        result.loc[end.gt(cutoff), "end_date"] = cutoff.strftime("%Y%m%d")
        if "ann_date" in result:
            announced = parse_date_series(result["ann_date"], errors="coerce")
            result = result.loc[announced.isna() | announced.le(cutoff)].copy()
    if filename == "stock_basic.parquet":
        delisted = parse_date_series(result["delist_date"], errors="coerce")
        future_delist = delisted.gt(cutoff)
        result.loc[future_delist, "delist_date"] = None
        result["list_status"] = "L"
        result.loc[delisted.le(cutoff), "list_status"] = "D"
    return result


def write_marker(config: dict, mode: str) -> None:
    marker = ROOT / config["data"]["window_marker"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "mode": mode,
                "design_end": config["windows"]["design_end"],
                "oos_start": config["windows"]["oos_start"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def seal(config: dict) -> None:
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    sealed_dir = ROOT / config["data"]["tushare_sealed_dir"]
    backup_dir = sealed_dir / "full_snapshot"
    backup_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(config["windows"]["design_end"])
    manifest = {"cutoff": str(cutoff.date()), "files": []}

    missing_sources = [
        str(raw_dir / name) for name in CORE_SPECS if not (raw_dir / name).exists()
    ]
    existing_backups = [
        str(backup_dir / name) for name in CORE_SPECS if (backup_dir / name).exists()
    ]
    if missing_sources:
        raise FileNotFoundError("封存前缺少核心文件：" + ", ".join(missing_sources))
    if existing_backups:
        raise FileExistsError("封存备份已存在：" + ", ".join(existing_backups))

    for filename in CORE_SPECS:
        source = raw_dir / filename
        backup = backup_dir / filename
        source.replace(backup)
        full = pd.read_parquet(backup)
        design = design_copy(filename, full, cutoff)
        atomic_parquet(design, source)
        manifest["files"].append(
            {
                "file": filename,
                "full_rows": len(full),
                "design_rows": len(design),
                "full_sha256": sha256(backup),
                "design_sha256": sha256(source),
            }
        )

    for source in sorted(raw_dir.glob("*.parquet")):
        match = re.fullmatch(r"(?:daily_basic|stk_limit|suspend_d)_(\d{4})\.parquet", source.name)
        if not match or int(match.group(1)) <= cutoff.year:
            continue
        destination = sealed_dir / source.name
        if destination.exists():
            raise FileExistsError(f"目标已存在，未覆盖：{destination}")
        source.replace(destination)

    (sealed_dir / "seal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_marker(config, "design")
    verify(config)
    print("OOS 数据已封存；当前模式：design")


def check_dates(path: Path, cutoff: pd.Timestamp) -> list[str]:
    errors = []
    frame = pd.read_parquet(path)
    date_columns = [
        column
        for column in ["trade_date", "cal_date", "start_date", "end_date", "ann_date", "list_date", "delist_date"]
        if column in frame.columns
    ]
    for column in date_columns:
        dates = parse_date_series(frame[column], errors="coerce")
        if dates.notna().any() and dates.max() > cutoff:
            errors.append(f"{path.relative_to(ROOT)}:{column} 最大日期 {dates.max().date()}")
    return errors


def check_direct_qlib_imports() -> list[str]:
    errors = []
    targets = (
        list((ROOT / "src").glob("*.py"))
        + list((ROOT / "scripts").glob("d2*.py"))
        + list((ROOT / "scripts").glob("d3*.py"))
        + list((ROOT / "scripts").glob("d4*.py"))
        + list((ROOT / "scripts").glob("d5*.py"))
        + list((ROOT / "scripts").glob("d6*.py"))
    )
    allowed = (ROOT / "src/data_access.py").resolve()
    for path in targets:
        if path.resolve() == allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "qlib.data":
                errors.append(f"{path.relative_to(ROOT)} 直接导入 qlib.data；必须走 ResearchDataAccess")
            if isinstance(node, ast.Import) and any(alias.name == "qlib.data" for alias in node.names):
                errors.append(f"{path.relative_to(ROOT)} 直接导入 qlib.data；必须走 ResearchDataAccess")
    return errors


def verify(config: dict) -> None:
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    processed_dir = ROOT / config["data"]["processed_design_dir"]
    cutoff = pd.Timestamp(config["windows"]["design_end"])
    errors = []
    for path in raw_dir.rglob("*.parquet"):
        errors.extend(check_dates(path, cutoff))
    if processed_dir.exists():
        for path in processed_dir.rglob("*.parquet"):
            errors.extend(check_dates(path, cutoff))
    errors.extend(check_direct_qlib_imports())

    access = ResearchDataAccess(ROOT)
    try:
        access.assert_window("2018-12-31", "2019-01-01")
    except WindowViolation:
        pass
    else:
        errors.append("窗口守卫没有拦截 2019-01-01")

    if errors:
        raise RuntimeError("封存验证失败：\n- " + "\n- ".join(errors))
    print("封存验证通过：项目数据与统一访问层均无法读取 2019-01-01 及以后日期。")


def unseal(config: dict, confirmation: str | None) -> None:
    if confirmation != "UNSEAL_OOS":
        raise ValueError("解封必须显式传入 --confirm UNSEAL_OOS")
    raw_dir = ROOT / config["data"]["tushare_raw_dir"]
    sealed_dir = ROOT / config["data"]["tushare_sealed_dir"]
    backup_dir = sealed_dir / "full_snapshot"
    missing_backups = [
        str(backup_dir / name) for name in CORE_SPECS if not (backup_dir / name).exists()
    ]
    if missing_backups:
        raise FileNotFoundError("解封前缺少备份：" + ", ".join(missing_backups))
    for filename in CORE_SPECS:
        backup = backup_dir / filename
        destination = raw_dir / filename
        if not backup.exists():
            raise FileNotFoundError(backup)
        destination.unlink(missing_ok=True)
        backup.replace(destination)
    for source in sorted(sealed_dir.glob("*.parquet")):
        if re.fullmatch(r"(?:daily_basic|stk_limit|suspend_d)_\d{4}\.parquet", source.name):
            destination = raw_dir / source.name
            if destination.exists():
                raise FileExistsError(destination)
            source.replace(destination)
    write_marker(config, "oos")
    print("OOS 数据已解封；当前模式：oos")


def main() -> None:
    parser = argparse.ArgumentParser(description="封存或解封 2019 年以后的数据")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seal")
    subparsers.add_parser("verify")
    unseal_parser = subparsers.add_parser("unseal")
    unseal_parser.add_argument("--confirm")
    args = parser.parse_args()
    config = load_config()
    if args.command == "seal":
        seal(config)
    elif args.command == "verify":
        verify(config)
    else:
        unseal(config, args.confirm)


if __name__ == "__main__":
    main()

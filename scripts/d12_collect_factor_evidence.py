"""Pull full-period independent adjustment factors for every security D11 ever priced.

Day 3 could only reach 0.62% independent close coverage and 0.00% independent
factor-ratio coverage, because the only caches on disk were D2's 20 sampled days
plus the two-code probe.  At that coverage a screen hit cannot be told apart from
a genuine ex-rights session, so no repair is defensible and none was made.

This script closes that gap and does nothing else.  It writes Tushare ``adj_factor``
and ``daily`` into a fresh evidence directory, one file per year, under names the
existing day-3 loader already recognises.  It never touches Qlib, the registered
caches, the D11 runs or the repair itself.

It is resumable: each security is cached individually, so an interrupted pull
restarts where it stopped rather than re-requesting what it already has.

Same upstream vendor as the project's own data, and that limit is stated in the
manifest.  What it is independent of is the *conversion chain* -- the Qlib
community bundle's factor series -- which is the chain under suspicion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repo import D12Error, sha256
from src.d2_core import to_tushare_symbol
from src.date_utils import parse_date_series

ENDPOINTS = {
    'adj_factor': {'fields': 'ts_code,trade_date,adj_factor', 'value': 'adj_factor'},
    'daily': {'fields': 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount', 'value': 'close'},
}


def request(method, delay, **kwargs):
    """The conservative retry policy the project's other Tushare pulls already use."""
    for attempt in range(6):
        try:
            time.sleep(delay)
            return method(**kwargs)
        except Exception as error:  # noqa: BLE001 - vendor client raises bare exceptions
            if attempt == 5:
                raise
            wait = min(60, 5 * 2 ** attempt)
            print(f'请求失败：{error}，{wait} 秒后重试', flush=True)
            time.sleep(wait)


def universe(config, root, end):
    """Every instrument that ever carried a benchmark weight, plus the alias codes."""
    names = set()
    for directory in [config['data']['processed_design_dir'], config['d9']['v2']['processed_oos_dir']]:
        path = root / directory / 'benchmark_weights.parquet'
        if not path.is_file():
            raise D12Error(f'Missing registered benchmark weights: {path}')
        frame = pd.read_parquet(path, columns=['trade_date', 'instrument'])
        frame = frame.loc[parse_date_series(frame.trade_date) <= pd.Timestamp(end)]
        names.update(frame.instrument.astype(str).str.upper())
    aliases = config['d9']['membership_recovery'].get('identifier_alias_removals', [])
    for rule in aliases:
        names.update({str(rule[key]).upper() for key in ['alias_instrument', 'canonical_instrument']})
    for rule in config['universe'].get('membership_replacements', []):
        names.update({str(rule[key]).upper() for key in ['remove_instrument', 'add_instrument']})
    return sorted(names)


def fetch_security(pro, code, endpoint, start, end, delay):
    spec = ENDPOINTS[endpoint]
    frame = request(getattr(pro, endpoint), delay, ts_code=code,
                    start_date=start, end_date=end, fields=spec['fields'])
    if frame is None or frame.empty:
        return pd.DataFrame(columns=spec['fields'].split(','))
    frame = frame.copy()
    frame['trade_date'] = frame.trade_date.astype(str)
    if not frame.trade_date.str.fullmatch(r'\d{8}').all():
        raise D12Error(f'{endpoint} returned an invalid trade_date for {code}')
    frame = frame.drop_duplicates().sort_values(['ts_code', 'trade_date'])
    if frame.duplicated(['ts_code', 'trade_date']).any():
        raise D12Error(f'{endpoint} returned conflicting duplicate rows for {code}')
    # Tushare caps a single call at a round number of rows and returns the cap
    # silently. A truncated history would look like "no evidence" for the missing
    # years, which is exactly the state this script exists to end.
    if len(frame) in {4000, 5000, 6000, 8000, 10000}:
        raise D12Error(f'{endpoint} returned exactly {len(frame)} rows for {code}; assume the call was '
                       'truncated and split the request by year before retrying')
    return frame.reset_index(drop=True)


def collect(pro, codes, endpoint, start, end, cache, delay):
    frames = []
    for position, code in enumerate(codes, 1):
        path = cache / f'{endpoint}_{code.replace(".", "_")}.parquet'
        if path.is_file():
            frames.append(pd.read_parquet(path))
            continue
        frame = fetch_security(pro, code, endpoint, start, end, delay)
        temporary = path.with_suffix('.parquet.tmp')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        frames.append(frame)
        if position % 25 == 0 or position == len(codes):
            print(f'  {endpoint}: {position}/{len(codes)} 只证券', flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='new immutable evidence run ID')
    parser.add_argument('--request-delay', type=float, default=0.4)
    parser.add_argument('--limit-instruments', type=int, help='smoke-test a prefix of the universe only')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run_id):
        parser.error('请使用简单且全新的 run-id')

    token = os.environ.get('QLIB_TUSHARE_TOKEN')
    if not token:
        raise D12Error('未设置环境变量 QLIB_TUSHARE_TOKEN')
    import tushare as ts

    config = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    start = pd.Timestamp(config['windows']['design_start'])
    end = pd.Timestamp(config['windows']['oos_end'])
    raw_dir = ROOT / config['data']['tushare_raw_dir'] / 'd12_evidence' / args.run_id
    report_dir = ROOT / 'reports/d12/evidence' / args.run_id
    # A *completed* run is immutable and its ID may never be reused. An interrupted
    # one has no manifest, and re-running the same ID picks up its per-security
    # cache instead of re-requesting a thousand securities from the vendor.
    if (report_dir / 'manifest.json').is_file():
        raise FileExistsError(f'{args.run_id} 已经完成，run-id 不可复用；请保留旧目录并使用新 ID')
    resuming = raw_dir.exists()
    cache = raw_dir / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if resuming:
        print(f'继续未完成的取证 {args.run_id}：已缓存 {len(list(cache.glob("*.parquet")))} 个证券文件', flush=True)

    names = universe(config, ROOT, end)
    if args.limit_instruments:
        names = names[:args.limit_instruments]
    codes = [to_tushare_symbol(name) for name in names]
    print(f'D12 证据：{len(codes)} 只证券，{start.date()} 至 {end.date()}', flush=True)
    pro = ts.pro_api(token)
    written, rows = {}, {}
    for endpoint in ENDPOINTS:
        frame = collect(pro, codes, endpoint, start.strftime('%Y%m%d'), end.strftime('%Y%m%d'),
                        cache, args.request_delay)
        rows[endpoint] = int(len(frame))
        if frame.empty:
            raise D12Error(f'Tushare {endpoint} 未返回任何数据')
        frame['year'] = frame.trade_date.str.slice(0, 4).astype(int)
        for year, group in frame.groupby('year'):
            # `scripts/d12_day3.load_reference` matches exactly this filename shape.
            path = raw_dir / f'{endpoint}_{year}.parquet'
            group.drop(columns='year').reset_index(drop=True).to_parquet(path, index=False)
            written[str(path.relative_to(ROOT))] = sha256(path)

    manifest = {
        'status': 'completed', 'diagnostic_only': True, 'qlib_data_changed': False, 'r0_repaired': False,
        'run_id': args.run_id, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'start_date': str(start.date()), 'end_date': str(end.date()),
        'instruments_requested': len(codes), 'rows': rows, 'files_sha256': written,
        'vendor_note': 'Same upstream vendor as the project data; independent of the Qlib conversion chain only.',
        'next_command': 'python -u scripts/d12_repair_r0.py --repair-id <id> --stage classify',
    }
    (report_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {report_dir} 与 {raw_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

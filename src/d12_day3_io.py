"""Immutable D12 day-3 outputs and source identity, without changing D11 guards."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.d12_repo import D12Error, sha256


def clean(value):
    from scripts.d11_diagnose import clean as original
    return original(value)


class Day3Output:
    def __init__(self, root, run_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
            raise D12Error('Use a simple new run ID')
        root = Path(root).resolve()
        parent = root / 'reports/d12/day3'
        if parent.resolve() != parent or (parent / run_id).is_symlink():
            raise D12Error('Output directory may not traverse symlinks')
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / run_id
        self.path.mkdir(exist_ok=False)

    def target(self, name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in {'.','..'}:
            raise D12Error('Output filename must stay inside run')
        p = self.path / name
        if p.exists() or p.is_symlink(): raise FileExistsError(p)
        return p

    def json(self, name, value):
        self.target(name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                               allow_nan=False, default=str)+'\n', encoding='utf-8')

    def text(self, name, value):
        self.target(name).write_text(value, encoding='utf-8')

    def parquet(self, name, frame):
        frame.to_parquet(self.target(name), index=False)

    def table(self, name, frame):
        self.text(name+'.tsv', frame.to_csv(sep='\t', index=False))

    def hashes(self):
        return {p.name: sha256(p) for p in sorted(self.path.iterdir()) if p.is_file() and p.name != 'manifest.json'}


def snapshot(root, config):
    """Same data/provider coverage as D11; exclude only immutable day-3 OUTPUT runs.

    D11's snapshot excludes its own reports/d11 runs. Day3 likewise excludes
    reports/d12/day3, otherwise its new output files would count as input changes.
    Old source-run manifests and every declared source output are checked separately.
    """
    root = Path(root).resolve()
    provider = (root / Path(config['data']['qlib_provider_uri']).expanduser()).resolve()
    paths = set()
    for directory in ['config','src','scripts','tests','reports','models','data']:
        for p in (root / directory).rglob('*'):
            rel = p.relative_to(root)
            if not p.is_file() or '__pycache__' in p.parts: continue
            if rel.parts[:2] == ('reports','d11') or rel.parts[:3] == ('reports','d12','day3'): continue
            paths.add(p)
    for directory in ['features','calendars','instruments']:
        paths.update(p for p in (provider / directory).rglob('*') if p.is_file())
    result = {}
    for i, p in enumerate(sorted(paths), 1):
        if i % 5000 == 0: print(f'D12 input SHA256 {i}/{len(paths)}', flush=True)
        key = str(p.relative_to(root)) if p.is_relative_to(root) else 'qlib/'+str(p.relative_to(provider))
        result[key] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    return result


def verify_manifest(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('inputs_unchanged') is not True:
        raise D12Error('Source run did not complete unchanged: '+str(path))
    for name, digest in manifest['output_sha256'].items():
        if Path(name).name != name or not (path/name).is_file() or sha256(path/name) != digest:
            raise D12Error('Source output missing/changed: '+str(path/name))
    return {'manifest_sha256': sha256(path/'manifest.json'), 'outputs_verified': len(manifest['output_sha256'])}


def source_input_parity(before, original):
    # New D12 code is legitimate; original data/models/provider and simulator are not replaceable.
    keys = [k for k in original if k.startswith(('data/','models/','qlib/')) or
            k in ['config/config.yaml','src/d11_core.py','scripts/d11_pipeline.py']]
    if not keys or not any(k.startswith(('data/','qlib/')) for k in keys):
        raise D12Error('Source input snapshot has no registered data/provider identities')
    changed = [k for k in keys if before.get(k) != original[k]]
    if changed: raise D12Error('Source data/model/simulator identity changed; establish a separately repaired R0: '+', '.join(changed[:20]))
    return {'passed': True, 'checked_source_entries': len(keys), 'new_d12_files_allowed': True}


def full_replay(market, saved, flags, calendar, config, output):
    """Original simulator and original check_parity tolerances; no impact estimator in decisions.

    This is a day-3 R0 reproduction. It is NOT the day-4 zero-coefficient test of
    a future modified R1 simulator, and it never certifies data quality.
    """
    from src.d11_core import simulate_causal
    from scripts.d11_diagnose import check_parity
    exposures = saved['factor_exposures']
    # D6 keeps two descriptive string columns beside the numeric one-hot matrix:
    # `industry_name` (for example "汽车") and `industry_factor`.  A prefix-only
    # scan accidentally handed both strings to CVXPY.  The actual D11 optimisation
    # columns are exactly the numeric `industry_*` columns written by
    # build_industry_dummies; retain all of them, including an all-zero column, so
    # the replay has the same constraint matrix as the saved run.
    industries = [c for c in exposures
                  if c.startswith('industry_') and c not in {'industry_name', 'industry_factor'}
                  and pd.api.types.is_numeric_dtype(exposures[c])]
    styles = ['style_'+s for s in config['d7']['constrained_style_factors']]
    missing_styles = [c for c in styles if c not in exposures]
    nonnumeric_styles = [c for c in styles if c in exposures
                         and not pd.api.types.is_numeric_dtype(exposures[c])]
    if not industries or missing_styles or nonnumeric_styles:
        raise D12Error('Saved factor schema cannot reproduce D11: '
                       f'industries={len(industries)}, missing_styles={missing_styles}, '
                       f'nonnumeric_styles={nonnumeric_styles}')
    if 'industry_factor' in exposures:
        declared = set(exposures['industry_factor'].dropna().astype(str).unique())
        if not declared.issubset(industries):
            raise D12Error('Saved industry_factor contains labels without numeric dummy columns: '
                           + ', '.join(sorted(declared-set(industries))))
    count = 0
    def progress(target, orders, audit):
        nonlocal count
        count += 1
        if count == 1 or count % 25 == 0: print(f'D12 R0 replay: {count}/{len(saved["audits"])} decisions', flush=True)
    result = simulate_causal(market, saved['predictions'], exposures, flags, calendar, config, industries, styles, progress=progress)
    daily = result.nav.copy()
    daily['portfolio_return'] = daily.nav/daily.nav.shift(1, fill_value=float(config['d8']['main_capital_cny']))-1
    if not daily.trade_date.equals(saved['daily'].trade_date): raise D12Error('Full replay date mismatch')
    errors = {
        'daily': check_parity(daily,saved['daily'],['trade_date'],['nav','cash','transaction_cost'],'baseline account',atol=1e-4),
        'returns': check_parity(daily,saved['daily'],['trade_date'],['single_sided_turnover','portfolio_return'],'baseline returns'),
        'targets': check_parity(result.targets,saved['targets'],['signal_date','instrument'],['target_weight','pretrade_weight'],'baseline targets'),
        'orders': check_parity(result.orders,saved['orders'],['signal_date','instrument'],['order_units','signal_nav'],'baseline orders',atol=1e-5),
        'trades': check_parity(result.trades,saved['trades'],['execution_date','instrument'],['executed_units','total_cost'],'baseline fills',atol=1e-5)}
    # Compare discrete policy/solver outcomes too; equal NAV alone is insufficient.
    for new, old, keys, cols in [
        (result.orders,saved['orders'],['signal_date','instrument'],['execution_date','signal_can_buy']),
        (result.trades,saved['trades'],['execution_date','instrument'],['exchange_can_buy','exchange_can_sell','participation_limited']),
        (result.audits,saved['audits'],['signal_date'],['solver_status','fallback_used','initial_construction_exempt'])]:
        a,b = new.set_index(keys).sort_index(),old.set_index(keys).sort_index()
        if not a.index.equals(b.index) or not a[cols].equals(b[cols]): raise D12Error('Replay discrete decisions changed')
    parity = {'passed': True, 'full_period': True, 'days': len(daily), 'decisions': len(result.audits),
              'maximum_errors': errors, 'tolerances_source': 'scripts/d11_diagnose.py::replay_stress/check_parity',
              'rtol': 2e-9, 'atol': {'daily':1e-4,'returns':1e-9,'targets':1e-9,'orders':1e-5,'trades':1e-5},
              'r1_zero_coefficient_gate_satisfied': False, 'p0_quality_certified': False}
    output.json('baseline_parity.json',parity)
    output.parquet('r0_replay_daily.parquet',daily)
    return parity

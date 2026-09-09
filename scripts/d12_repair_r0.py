"""Repair the Qlib factor chain, invalidate what depends on it, and rebuild R0.

Day 3 established the defect and stopped there on purpose: with 0.62% independent
price coverage, "this adjusted return is impossible" was an assertion, not a
finding.  This entry point closes the remaining day-3 items in the order the plan
fixes -- evidence, then repair, then invalidation, then a rebuilt R0 -- and refuses
to skip a step.

    --stage classify   independent evidence -> per-session verdicts -> frozen repair plan
    --stage provider   a corrected Qlib bundle beside the registered one, byte-verified
    --stage inputs     rebuild the caches that read prices; prove the design window is untouched
    --stage r0         D11's own research pipeline, unchanged, on the corrected inputs
    --stage compare    original vs repaired, every metric side by side

What it does not do.  It does not repair index membership: P0-12 is still blocked on
official effective-date constituent sets, so the repaired R0 carries the same
membership timing as the original and the same disclosed A-class gap.  Changing one
mechanism at a time is the only way the difference between the two runs stays
attributable.  It does not overwrite `reports/d11`, does not touch the registered
provider, and does not relax a D11 guard.

A repaired R0 is a *new* baseline, not a better result.  Its return may go up or
down; both are reported, and the historical No-Go stands either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repo import D12Error, sha256
from src.d12_repair_io import StageOutput, require_stage
from src.d12_price_repair import (build_repair_plan, classification_summary, classify_factor_rows,
                                  repair_preview, resolve_reference_duplicates, verify_repair)
from src.d12_provider import (build_corrected_provider, instrument_directory, prove_bin_decoding,
                              read_provider_calendar, verify_corrected_provider)
from src.d12_repair_access import RepairedDataAccess, repaired_config
from src.d12_day3 import price_audit, reference_crosscheck
from src.d2_core import to_qlib_symbol
from src.d11_io import D11Output
from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series
from src.d9_core import summarize_monthly_performance
from scripts.d11_pipeline import load_open_calendar
from scripts.d3_pipeline import load_qlib_market
from scripts.d12_day3 import load_limits, load_weights_flags
from scripts import d12_preflight as preflight

PROVIDER_PARENT = 'data/d12_repair'
STAGES = ('classify', 'provider', 'inputs', 'r0', 'compare')


def say(message):
    print(message, flush=True)


def load_config():
    return yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))


def run_preflight(output, config):
    """The same conditions D11's preflight enforces, from D12's own entry point."""
    report = preflight.Report()
    for check, rest in [(preflight.check_repository, [json.loads((ROOT / 'config/d11_baseline.json').read_text())]),
                        (preflight.check_inputs, [config]), (preflight.check_registered_digests, []),
                        (preflight.check_environment, [config]), (preflight.check_saved_runs, [])]:
        check(report, *rest)
    output.json('preflight.json', {'stages': report.stages})
    if report.blocking:
        raise D12Error('Repair preflight blocked; see preflight.json')
    return report.stages[0]['detail']


# --------------------------------------------------------------------------- classify

#: Everything the repair plan, its verification and a manual review actually read.
#: The full audit is ~40 columns over millions of sessions; keeping it whole would
#: cost gigabytes to answer questions none of the later stages ask.
RETAINED = ['trade_date', 'instrument', 'factor', 'previous_factor', 'previous_trade_date',
            'reference_factor', 'previous_reference_factor', 'observed_factor_ratio',
            'independent_factor_ratio', 'factor_ratio_error', 'implied_correction',
            'raw_close', 'reference_close', 'raw_close_error', 'raw_close_contradicted',
            'mark_return', 'raw_return_1d', 'return_span_sessions', 'limit_status',
            'up_limit', 'down_limit', 'weight', 'in_project_universe', 'verdict']


def load_full_reference(access, config, start, end):
    """Every independent daily/adj_factor cache on disk, merged across evidence runs.

    ``scripts.d12_day3.load_reference`` refuses any repeated ``(session, security)``
    outright, which is the right rule for an immutable audit run reading one cache.
    A repair reads several -- the narrow two-code probe and the full-period pull
    overlap by construction -- so identical repeats are collapsed here and only
    genuine disagreements are raised.
    """
    raw = access.root / config['data']['tushare_raw_dir']
    parts = {'daily': [], 'adj_factor': []}
    inventory = []
    for path in sorted(raw.rglob('*.parquet')):
        match = re.fullmatch(r'(daily|adj_factor)_(\d{4}|\d{8})\.parquet', path.name)
        if not match:
            continue
        endpoint = match.group(1)
        frame = access.read_parquet(path.relative_to(access.root), date_col='trade_date',
                                    start_date=start, end_date=end)
        frame['trade_date'] = parse_date_series(frame.trade_date)
        frame['instrument'] = frame.ts_code.map(to_qlib_symbol)
        column = 'close' if endpoint == 'daily' else 'adj_factor'
        renamed = 'reference_close' if endpoint == 'daily' else 'reference_factor'
        parts[endpoint].append(frame[['trade_date', 'instrument', column]].rename(columns={column: renamed}))
        inventory.append({'file': str(path.relative_to(access.root)), 'endpoint': endpoint,
                          'sha256': sha256(path), 'rows_in_scope': int(len(frame))})
    frames = []
    for endpoint, column in [('daily', 'reference_close'), ('adj_factor', 'reference_factor')]:
        if parts[endpoint]:
            frames.append(resolve_reference_duplicates(pd.concat(parts[endpoint], ignore_index=True), column))
        else:
            frames.append(pd.DataFrame({'trade_date': pd.Series(dtype='datetime64[ns]'),
                                        'instrument': pd.Series(dtype=str), column: pd.Series(dtype=float)}))
    merged = frames[0].merge(frames[1], on=['trade_date', 'instrument'], how='outer', validate='one_to_one')
    return merged, inventory


def scan_for_classification(access, config, weights, calendar, output, chunk_size):
    """Re-screen every priced session, now against full-period independent factors.

    Rows are counted in full but only kept where they can still matter: every session
    of any instrument that has a contradiction, plus every session anywhere that no
    independent factor could resolve.  Sessions that agree with the evidence on an
    instrument with nothing to repair are counted and dropped.
    """
    start, end = pd.Timestamp(config['windows']['design_start']), pd.Timestamp(config['windows']['oos_end'])
    dates = calendar[(calendar >= start) & (calendar <= end)]
    aliases = config['d9']['membership_recovery'].get('identifier_alias_removals', [])
    alias_names = {rule[key] for rule in aliases for key in ['alias_instrument', 'canonical_instrument']}
    names = sorted(set(weights.instrument) | alias_names)
    limits, missing_limits = load_limits(access, config, start, end)
    reference, inventory = load_full_reference(access, config, start, end)
    output.json('independent_reference_inventory.json', {
        'files': inventory, 'missing_limit_years': missing_limits,
        'note': 'Coverage is measured, never assumed; an uncovered session stays unresolved.'})
    retained, totals = [], []
    for position in range(0, len(names), chunk_size):
        batch = names[position:position + chunk_size]
        say(f'D12 repair screen: {min(position + chunk_size, len(names))}/{len(names)} instruments')
        raw = load_qlib_market(access, batch, start, end,
                               float(config['tradability']['qlib_amount_multiplier_to_cny']))
        if raw.empty:
            continue
        audit = price_audit(raw, limits.loc[limits.instrument.isin(batch)], dates)
        audit = reference_crosscheck(audit, reference.loc[reference.instrument.isin(batch)])
        audit = audit.merge(weights[['trade_date', 'instrument', 'weight']],
                            on=['trade_date', 'instrument'], how='left', validate='one_to_one')
        audit['in_project_universe'] = audit.weight.notna()
        classified = classify_factor_rows(audit)[RETAINED]
        totals.append(classification_summary(classified))
        suspect = classified.instrument.isin(
            classified.loc[classified.verdict.eq('factor_contradicted') | classified.raw_close_contradicted,
                           'instrument'].unique())
        retained.append(classified.loc[suspect | classified.verdict.eq('no_independent_evidence')].copy())
    if not totals:
        raise D12Error('The repair screen priced no instruments at all')
    counts = pd.DataFrame(totals).drop(columns='independent_coverage').sum().astype(int).to_dict()
    counts['independent_coverage'] = float(
        (counts['observed_rows'] - counts['no_independent_evidence']) / max(1, counts['observed_rows']))
    return pd.concat(retained, ignore_index=True), counts


def stage_classify(args, config, output):
    repository = run_preflight(output, config)
    access = ResearchDataAccess(ROOT)
    if access.mode != 'oos':
        raise D12Error('Original OOS marker required; this entry point never unseals data')
    start, end = pd.Timestamp(config['windows']['design_start']), pd.Timestamp(config['windows']['oos_end'])
    calendar = load_open_calendar(access, config, end_date=end)
    weights, _ = load_weights_flags(access, config, start, end)
    classified, summary = scan_for_classification(access, config, weights, calendar, output, args.chunk_size)
    say(f'D12 verdicts: {summary}')
    # Write the evidence before testing it. A coverage failure is the most likely way
    # this stage stops, and it is only actionable if the operator can see *which*
    # securities and years came back uncovered.
    contradicted = classified.loc[classified.verdict.eq('factor_contradicted')].copy()
    output.parquet('factor_classification_retained.parquet', classified)
    output.table('factor_contradictions', contradicted.sort_values(['instrument', 'trade_date']))
    unresolved = classified.loc[classified.verdict.eq('no_independent_evidence') & classified.in_project_universe]
    output.table('unresolved_sessions_top', unresolved.assign(
        magnitude=(unresolved.mark_return).abs()).sort_values('magnitude', ascending=False).head(500))
    if len(unresolved):
        output.table('unresolved_by_year', unresolved.groupby(unresolved.trade_date.dt.year).agg(
            unresolved_rows=('instrument', 'size'), instruments=('instrument', 'nunique')).reset_index())
        output.table('unresolved_by_instrument', unresolved.groupby('instrument').agg(
            unresolved_rows=('trade_date', 'size'), first_session=('trade_date', 'min'),
            last_session=('trade_date', 'max')).reset_index().sort_values('unresolved_rows', ascending=False))
    if summary['independent_coverage'] < args.minimum_coverage:
        raise D12Error(
            f'Independent factor coverage is {summary["independent_coverage"]:.2%}, below the required '
            f'{args.minimum_coverage:.2%}. Run scripts/d12_collect_factor_evidence.py first; a repair '
            'planned on thin evidence is a guess. See unresolved_by_year.tsv and unresolved_by_instrument.tsv '
            'in this stage directory for what is still missing.')
    plan = build_repair_plan(classified, anchor=args.anchor)
    proof = verify_repair(classified, plan)
    preview = repair_preview(classified, plan)
    design_end = pd.Timestamp(config['windows']['design_end'])
    design_breaks = plan.loc[plan.breakpoint_date.notna() & plan.breakpoint_date.le(design_end)]
    output.parquet('repair_plan.parquet', plan)
    output.table('repair_plan', plan)
    output.table('repair_preview_returns', preview.loc[
        preview.adjusted_return_change.abs().gt(1e-9) | preview.verdict.eq('factor_contradicted')]
        .sort_values(['instrument', 'trade_date']))
    output.json('repair_plan.json', {
        'anchor_rule': args.anchor, 'segments': plan.to_dict('records'),
        'verification': proof, 'classification': summary,
        'membership_repair': 'not attempted; P0-12 stays blocked on official effective-date sets',
        'raw_price_defects_excluded': int(classified.raw_close_contradicted.sum()),
        'design_window_breakpoints': design_breaks.to_dict('records'),
        'design_window_note': 'A breakpoint at or before design_end would move frozen design weights; '
                              'D9 re-asserts those at rtol=1e-12, so the inputs stage refuses to proceed '
                              'and the design caches would have to be rebuilt as a separate recorded decision.'})
    say(f'D12 repair plan: {len(plan)} segment(s) over {plan.instrument.nunique() if len(plan) else 0} instrument(s)')
    if len(design_breaks):
        say(f'D12 WARNING: {len(design_breaks)} breakpoint(s) fall inside the design window; --stage inputs will stop')
    return {'status': 'completed', 'repository': repository, 'classification': summary,
            'plan_segments': int(len(plan)), 'plan_instruments': int(plan.instrument.nunique()) if len(plan) else 0,
            'design_window_breakpoints': int(len(design_breaks)),
            'verification': proof, 'anchor_rule': args.anchor}


# --------------------------------------------------------------------------- provider

def stage_provider(args, config, output):
    run_preflight(output, config)
    classify_dir, classify_manifest = require_stage(ROOT, args.repair_id, 'classify')
    plan = pd.read_parquet(classify_dir / 'repair_plan.parquet')
    if plan.empty:
        raise D12Error('The frozen plan is empty; there is nothing to repair and no corrected provider to build')
    source = Path(config['data']['qlib_provider_uri']).expanduser().resolve()
    target = ROOT / PROVIDER_PARENT / args.repair_id / 'provider'
    calendar = read_provider_calendar(source)

    access = ResearchDataAccess(ROOT)
    end = pd.Timestamp(config['windows']['oos_end'])

    def reader(instrument, field, first, last):
        served = access.features([instrument], [f'${field}'], first, last)
        return served.reset_index().set_index('datetime')[f'${field}']

    samples = [(instrument, field) for instrument in sorted(plan.instrument.unique())[:3]
               for field in ['close', 'factor'] if (instrument_directory(source, instrument) / f'{field}.day.bin').is_file()]
    proof = prove_bin_decoding(source, calendar, samples, reader)
    output.json('bin_decoding_proof.json', {'checks': proof, 'end_of_window': str(end.date())})
    say(f'D12 bin decoding proven on {len(proof)} instrument/field pair(s)')

    patches, provider_summary = build_corrected_provider(source, target, plan, mode=args.mirror, progress=say)
    verification = verify_corrected_provider(source, target, plan, patches, progress=say)
    output.table('provider_patches', patches)
    output.json('provider_verification.json', {**provider_summary, **verification,
                                               'classify_manifest_sha256': sha256(classify_dir / 'manifest.json')})
    say(f'D12 corrected provider: {provider_summary} / {verification}')
    return {'status': 'completed', 'corrected_provider': str(target), 'source_provider': str(source),
            'provider': provider_summary, 'verification': verification,
            'classify_stage_sha256': classify_manifest['output_sha256']}


# --------------------------------------------------------------------------- inputs

def stage_inputs(args, config, output):
    # Re-run the full preflight rather than trusting the earlier stage: hours can pass
    # between stages, and the 140 protected files must still be intact right now.
    run_preflight(output, config)
    classify_dir, classify_manifest = require_stage(ROOT, args.repair_id, 'classify')
    _, provider_manifest = require_stage(ROOT, args.repair_id, 'provider')
    if classify_manifest.get('design_window_breakpoints'):
        raise D12Error(
            f'{classify_manifest["design_window_breakpoints"]} repair breakpoint(s) fall at or before '
            f'{config["windows"]["design_end"]}. Rebuilding OOS caches alone would leave the frozen design '
            'weights inconsistent with the corrected prices, and D9 re-asserts those at rtol=1e-12. Rebuilding '
            'the D2 design caches is a separate, recorded decision; see repair_plan.json '
            f'({classify_dir.relative_to(ROOT)}/repair_plan.json).')
    provider = Path(provider_manifest['corrected_provider'])
    access = RepairedDataAccess(ROOT, provider)
    from scripts.d9_pipeline import build_full_benchmark_weights, build_oos_tradability

    end = pd.Timestamp(config['windows']['oos_end'])
    oos_start = pd.Timestamp(config['windows']['oos_start'])
    design_end = pd.Timestamp(config['windows']['design_end'])
    calendar = load_open_calendar(access, config, end_date=end)
    say('D12 rebuilding PIT benchmark weights from the corrected provider')
    # The registered design directory stays in this config on purpose: the parity
    # assertion inside must compare against the ORIGINAL design weights, so a repair
    # that reaches back into the design window fails here instead of passing quietly.
    weights, weight_audit = build_full_benchmark_weights(access, config, calendar)
    say('D12 rebuilding OOS tradability flags from the corrected provider')
    tradability, tradability_audit = build_oos_tradability(access, config, weights, calendar)

    processed = ROOT / PROVIDER_PARENT / args.repair_id / 'processed'
    design_dir, oos_dir = processed / 'design', processed / 'oos'
    if processed.exists():
        raise D12Error(f'Rebuilt cache directory already exists; use a new repair ID: {processed}')
    design_dir.mkdir(parents=True)
    oos_dir.mkdir(parents=True)
    copied = {}
    for name in ['benchmark_weights', 'tradability_flags']:
        origin = ROOT / config['data']['processed_design_dir'] / f'{name}.parquet'
        shutil.copy2(origin, design_dir / f'{name}.parquet')
        copied[name] = {'source': str(origin.relative_to(ROOT)), 'sha256': sha256(design_dir / f'{name}.parquet')}
        if copied[name]['sha256'] != sha256(origin):
            raise D12Error(f'Design cache copy does not match the registered file: {name}')
    weights.loc[weights.trade_date.between(oos_start, end)].to_parquet(oos_dir / 'benchmark_weights.parquet', index=False)
    tradability.to_parquet(oos_dir / 'tradability_flags.parquet', index=False)

    registered = pd.read_parquet(ROOT / config['d9']['v2']['processed_oos_dir'] / 'benchmark_weights.parquet')
    registered['trade_date'] = parse_date_series(registered.trade_date)
    rebuilt = weights.loc[weights.trade_date.between(oos_start, end), ['trade_date', 'instrument', 'weight']]
    difference = registered[['trade_date', 'instrument', 'weight']].merge(
        rebuilt, on=['trade_date', 'instrument'], how='outer', suffixes=('_original', '_repaired'), indicator=True)
    difference['weight_change'] = difference.weight_repaired - difference.weight_original
    membership_changed = difference['_merge'].ne('both')
    moved = difference.loc[membership_changed | difference.weight_change.abs().gt(1e-12)].copy()
    by_day = moved.groupby('trade_date').agg(
        rows=('instrument', 'size'), absolute_change=('weight_change', lambda s: float(s.abs().sum())),
        maximum_change=('weight_change', lambda s: float(s.abs().max()))).reset_index()
    output.table('benchmark_weight_changes_by_day', by_day.sort_values('absolute_change', ascending=False))
    output.table('benchmark_weight_changes_top', moved.reindex(
        moved.weight_change.abs().sort_values(ascending=False).index).head(500))
    output.json('rebuilt_inputs.json', {
        'processed_design_dir': str(design_dir.relative_to(ROOT)), 'processed_oos_dir': str(oos_dir.relative_to(ROOT)),
        'design_files_copied_unchanged': copied, 'design_window_parity': 'asserted by build_full_benchmark_weights',
        'design_end': str(design_end.date()),
        'benchmark_weight_audit': weight_audit, 'tradability_audit': tradability_audit,
        'oos_weight_rows_changed': int(len(moved)), 'oos_weight_days_changed': int(moved.trade_date.nunique()),
        'maximum_weight_change': float(moved.weight_change.abs().max()) if len(moved) else 0.0,
        'membership_unchanged': not bool(membership_changed.any()),
        'sha256': {'oos/benchmark_weights.parquet': sha256(oos_dir / 'benchmark_weights.parquet'),
                   'oos/tradability_flags.parquet': sha256(oos_dir / 'tradability_flags.parquet')}})
    say(f'D12 rebuilt weights: {len(moved)} row(s) on {moved.trade_date.nunique()} day(s) moved')
    return {'status': 'completed', 'processed_design_dir': str(design_dir), 'processed_oos_dir': str(oos_dir),
            'corrected_provider': str(provider), 'oos_weight_rows_changed': int(len(moved)),
            'oos_weight_days_changed': int(moved.trade_date.nunique()),
            'membership_unchanged': not bool(membership_changed.any())}


# --------------------------------------------------------------------------- r0

def stage_r0(args, config, output):
    run_preflight(output, config)
    _, inputs_manifest = require_stage(ROOT, args.repair_id, 'inputs')
    from scripts.d11_pipeline import run_research

    provider = Path(inputs_manifest['corrected_provider'])
    access = RepairedDataAccess(ROOT, provider)
    repaired = repaired_config(
        config,
        processed_design_dir=Path(inputs_manifest['processed_design_dir']).relative_to(ROOT),
        processed_oos_dir=Path(inputs_manifest['processed_oos_dir']).relative_to(ROOT))
    run = D11Output(ROOT, args.r0_run_id) if args.r0_in_d11_namespace else StageOutput(
        ROOT, args.repair_id, args.r0_run_id, parent='reports/d12/r0_fixed')
    say(f'D12 running the unmodified D11 research pipeline into {run.path}')
    began = time.perf_counter()
    summary = run_research(access, repaired, run)
    elapsed = time.perf_counter() - began
    # Hash the run before its own manifest lands in it, so the manifest describes a
    # closed set of files the way `d11_v1/manifest.json` does.
    digests = {p.name: sha256(p) for p in sorted(run.path.iterdir()) if p.is_file()}
    payload = {'status': 'completed', 'classification': 'repaired_r0_after_factor_chain_fix',
               'historical_decision': 'No-Go', 'new_oos_validation': False,
               'source_of_truth_for_comparison': 'reports/d11/d11_v1',
               'corrected_provider': str(provider), 'processed_oos_dir': inputs_manifest['processed_oos_dir'],
               'membership_repair': 'not attempted; P0-12 remains blocked',
               'elapsed_seconds': elapsed, 'research': summary,
               'outputs': len(digests), 'output_sha256': digests}
    run.json('repaired_r0_manifest.json', payload)
    output.json('r0_run.json', {**payload, 'run_directory': str(run.path.relative_to(ROOT))})
    say(f'D12 repaired R0 finished in {elapsed / 60:.1f} min: {summary.get("performance")}')
    return {'status': 'completed', 'run_directory': str(run.path), 'elapsed_seconds': elapsed,
            'outputs': len(digests), 'performance': summary.get('performance')}


# --------------------------------------------------------------------------- compare

def stage_compare(args, config, output):
    _, r0_manifest = require_stage(ROOT, args.repair_id, 'r0')
    repaired_run = Path(r0_manifest['run_directory'])
    original_run = ROOT / 'reports/d11/d11_v1'
    frames, performance, monthly = {}, {}, {}
    for label, path in [('original', original_run), ('repaired', repaired_run)]:
        daily = pd.read_parquet(path / 'daily.parquet')
        daily['trade_date'] = parse_date_series(daily.trade_date)
        frames[label] = daily
        performance[label], monthly[label] = summarize_monthly_performance(
            daily, newey_west_lags=config['d9']['newey_west_lags'])
    joined = monthly['original'][['month', 'excess_return']].rename(columns={'excess_return': 'original'}).merge(
        monthly['repaired'][['month', 'excess_return']].rename(columns={'excess_return': 'repaired'}),
        on='month', how='outer').sort_values('month')
    joined['difference'] = joined.repaired - joined.original
    output.table('monthly_excess_comparison', joined)
    rows = []
    for metric in sorted(set(performance['original']) | set(performance['repaired'])):
        left, right = performance['original'].get(metric), performance['repaired'].get(metric)
        rows.append({'metric': metric, 'original_d11_v1': left, 'repaired_r0': right,
                     'difference': (right - left) if isinstance(left, (int, float)) and isinstance(right, (int, float))
                     and not isinstance(left, bool) and not isinstance(right, bool) else None})
    comparison = pd.DataFrame(rows)
    output.table('results_comparison', comparison)
    daily_join = frames['original'][['trade_date', 'nav']].rename(columns={'nav': 'original_nav'}).merge(
        frames['repaired'][['trade_date', 'nav']].rename(columns={'nav': 'repaired_nav'}),
        on='trade_date', how='outer').sort_values('trade_date')
    daily_join['nav_difference'] = daily_join.repaired_nav - daily_join.original_nav
    first_divergence = daily_join.loc[daily_join.nav_difference.abs().gt(1e-6), 'trade_date']
    output.table('daily_nav_comparison', daily_join)
    payload = {
        'original_run': str(original_run.relative_to(ROOT)), 'repaired_run': str(repaired_run),
        'original_performance': performance['original'], 'repaired_performance': performance['repaired'],
        'first_nav_divergence': str(first_divergence.min().date()) if len(first_divergence) else None,
        'diverging_sessions': int(len(first_divergence)),
        'maximum_absolute_nav_difference': float(daily_join.nav_difference.abs().max()),
        'months_compared': int(joined.month.nunique()),
        'largest_monthly_difference_pp': float(joined.difference.abs().max() * 100) if joined.difference.notna().any() else 0.0,
        'historical_decision': 'No-Go', 'new_oos_validation': False,
        'reading': 'A repaired baseline replaces the old one for future comparison. The change is a data '
                   'correction, not a strategy improvement, and must never be reported as one.',
        'still_open': ['P0-12 index membership effective dates and official weights remain blocked',
                       'attribution quality was not re-certified by this run',
                       'the day-4 zero-coefficient R1 gate is separate and still required'],
    }
    output.json('comparison.json', payload)
    say(f'D12 comparison: {payload["original_performance"]} -> {payload["repaired_performance"]}')
    return {'status': 'completed', **payload}


# --------------------------------------------------------------------------- entry

RUNNERS = {'classify': stage_classify, 'provider': stage_provider, 'inputs': stage_inputs,
           'r0': stage_r0, 'compare': stage_compare}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repair-id', required=True, help='new immutable repair ID')
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--chunk-size', type=int, default=80, help='price loading memory bound; not a scope change')
    parser.add_argument('--anchor', default='earliest', choices=['earliest', 'latest', 'largest'],
                        help='which factor segment keeps multiplier 1; cannot change any return, but only '
                             '"earliest" leaves the frozen design window bit-identical')
    parser.add_argument('--minimum-coverage', type=float, default=.90,
                        help='required independent factor-ratio coverage before a plan may be frozen')
    parser.add_argument('--mirror', default='link', choices=['link', 'copy'],
                        help='hard-link the untouched provider files, or copy them')
    parser.add_argument('--r0-run-id', default='r0_fixed_v1', help='run ID for the repaired R0')
    parser.add_argument('--r0-in-d11-namespace', action='store_true',
                        help='write the repaired R0 under reports/d11/<id> instead of reports/d12/r0_fixed')
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    if not 0 < args.minimum_coverage <= 1:
        parser.error('--minimum-coverage must be a fraction in (0, 1]')

    config = load_config()
    output = StageOutput(ROOT, args.repair_id, args.stage)
    began = time.perf_counter()
    try:
        manifest = RUNNERS[args.stage](args, config, output)
    except Exception as error:  # noqa: BLE001 - every failure must leave evidence behind
        manifest = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        output.text('failure.txt', traceback.format_exc())
        print(manifest['error'], file=sys.stderr, flush=True)
    manifest.update(recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    elapsed_seconds=time.perf_counter() - began, argv=sys.argv[1:])
    output.finish(manifest)
    say(f'Wrote {output.path}; status={manifest["status"]}')
    return 0 if manifest['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

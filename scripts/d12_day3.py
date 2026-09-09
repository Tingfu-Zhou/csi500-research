"""Day 3: P0 price/membership evidence, P1, and optional full original-R0 replay.

Run on the original computer. Never fetches/repairs data, trains models, edits D11,
changes frozen R1 parameters, or starts a nonzero R1 experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_repo import D12Error, sha256
from src.d12_day3_io import Day3Output, full_replay, snapshot, source_input_parity, verify_manifest
from src.d12_day3 import (price_audit, reference_crosscheck, price_propagation, membership_timing,
                          suspected_membership_windows, official_membership_comparison, trade_groups)
from src.d12_core import ImpactEstimator, rolling_estimates
from src.d11_io import assert_snapshot_unchanged
from src.d11_diagnostics import reconstruct_d11
from src.d3_core import CostModel
from src.d2_core import to_qlib_symbol
from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series
from scripts.d11_pipeline import load_open_calendar, prepare_oos_market, read_frame, verify_run
from scripts.d11_diagnose import read_saved
from scripts.d3_pipeline import load_qlib_market
from scripts import d12_preflight as preflight

SAVED = ['daily','trades','orders','targets','audits','predictions',
         'labels_for_training_and_evaluation','factor_exposures']


def load_reference(access, config, start, end):
    """Read ONLY existing independent daily/adj_factor caches; report exact coverage.

    D2 normally cached just 20 sampled days under qc_sample. That is not a full
    independent audit. Absence remains visible; this entry point has no Tushare API.
    """
    raw = access.root / config['data']['tushare_raw_dir']
    parts = {'daily':[], 'adj_factor':[]}
    inventory = []
    for p in sorted(raw.rglob('*.parquet')):
        match = re.fullmatch(r'(daily|adj_factor)_(\d{4}|\d{8})\.parquet',p.name)
        if not match: continue
        endpoint = match.group(1)
        frame = access.read_parquet(p.relative_to(access.root), date_col='trade_date', start_date=start, end_date=end)
        frame['trade_date'] = parse_date_series(frame.trade_date)
        frame['instrument'] = frame.ts_code.map(to_qlib_symbol)
        col = 'close' if endpoint == 'daily' else 'adj_factor'
        renamed = 'reference_close' if endpoint == 'daily' else 'reference_factor'
        parts[endpoint].append(frame[['trade_date','instrument',col]].rename(columns={col:renamed}))
        inventory.append({'file':str(p.relative_to(access.root)), 'sha256':sha256(p), 'rows_in_scope':len(frame)})
    frames = []
    for endpoint, col in [('daily','reference_close'),('adj_factor','reference_factor')]:
        if parts[endpoint]:
            frame = pd.concat(parts[endpoint],ignore_index=True).drop_duplicates()
            if frame.duplicated(['trade_date','instrument']).any(): raise D12Error('Conflicting independent reference caches: '+endpoint)
            frames.append(frame)
        else:
            frames.append(pd.DataFrame({'trade_date':pd.Series(dtype='datetime64[ns]'),
                                        'instrument':pd.Series(dtype=str),col:pd.Series(dtype=float)}))
    return frames[0].merge(frames[1],on=['trade_date','instrument'],how='outer',validate='one_to_one'), inventory


def load_limits(access, config, start, end):
    frames, missing = [], []
    for year in range(start.year,end.year+1):
        p = Path(config['data']['tushare_raw_dir'])/f'stk_limit_{year}.parquet'
        if not (access.root/p).is_file(): missing.append(str(p)); continue
        frame = access.read_parquet(p,date_col='trade_date',start_date=start,end_date=end)
        frame['trade_date'] = parse_date_series(frame.trade_date)
        frame['instrument'] = frame.ts_code.map(to_qlib_symbol)
        frames.append(frame[['trade_date','instrument','up_limit','down_limit']])
    if not frames: raise D12Error('No existing stk_limit input')
    return pd.concat(frames,ignore_index=True),missing


def scan_prices(access, config, weights, calendar, output, chunk_size):
    start,end = pd.Timestamp(config['windows']['design_start']),pd.Timestamp(config['windows']['oos_end'])
    dates = calendar[(calendar>=start)&(calendar<=end)]
    aliases = config['d9']['membership_recovery'].get('identifier_alias_removals',[])
    alias_names = {r[k] for r in aliases for k in ['alias_instrument','canonical_instrument']}
    names = sorted(set(weights.instrument)|alias_names)
    limits, missing_limits = load_limits(access,config,start,end)
    reference, inventory = load_reference(access,config,start,end)
    output.json('independent_reference_inventory.json',{'files':inventory,'missing_limit_years':missing_limits,
        'note':'Missing/zero limit rows do not prove unlimited trading; reference caches may be sampled only.'})
    candidates, focus, coverage = [], [], []
    for i in range(0,len(names),chunk_size):
        batch = names[i:i+chunk_size]
        print(f'D12 P0 prices: {min(i+chunk_size,len(names))}/{len(names)} instruments, full history',flush=True)
        raw = load_qlib_market(access,batch,start,end,float(config['tradability']['qlib_amount_multiplier_to_cny']))
        observed_names = set(raw.instrument)
        absent_names = set(batch)-observed_names
        # Preserve completely absent aliases so missing source rows remain observable.
        if absent_names:
            raw = pd.concat([raw,pd.DataFrame({'instrument':sorted(absent_names),'trade_date':dates[0]})],ignore_index=True)
        audit = price_audit(raw,limits.loc[limits.instrument.isin(batch)],dates)
        audit = reference_crosscheck(audit,reference.loc[reference.instrument.isin(batch)])
        audit = audit.merge(weights[['trade_date','instrument','weight']],on=['trade_date','instrument'],how='left',validate='one_to_one')
        audit['in_project_universe'] = audit.weight.notna()
        candidates.append(audit.loc[audit.review_candidate].copy())
        # Both code series across their entire available history; only rows with data or a live project weight.
        focus.append(audit.loc[audit.instrument.isin(alias_names) & (audit.observed_price | audit.in_project_universe)].copy())
        for n,g in audit.groupby('instrument'):
            live = g.loc[g.in_project_universe]
            coverage.append({'instrument':n,'calendar_rows':len(g),'observed_prices':int(g.observed_price.sum()),
                'project_universe_rows':len(live),'missing_price_in_universe':int((~live.observed_price).sum()),
                'missing_mark_in_universe':int(live.mark.isna().sum()),
                'reference_close_rows':int(live.reference_close.notna().sum()),
                'reference_factor_ratio_rows':int(live.reference_factor_ratio.notna().sum()),
                'limit_valid_rows':int(live.limit_status.eq('valid').sum()),
                'limit_missing_rows':int(live.limit_status.eq('missing').sum()),
                'limit_zero_rows':int(live.limit_status.eq('zero_placeholder').sum()),
                'limit_invalid_rows':int(live.limit_status.eq('invalid').sum()),
                'candidate_rows_all_history':int(g.review_candidate.sum()),
                'candidate_rows_in_universe':int(live.review_candidate.sum())})
    candidates = pd.concat(candidates,ignore_index=True)
    output.parquet('price_candidates.parquet',candidates)
    output.table('price_coverage',pd.DataFrame(coverage))
    ranked = candidates.assign(abs_return=candidates.mark_return.abs()).sort_values(
        ['in_project_universe','adjusted_jump_candidate','abs_return'],ascending=False)
    output.table('price_candidates_top',ranked.head(500))
    alias = pd.concat(focus,ignore_index=True)
    output.parquet('alias_price_history.parquet',alias)
    output.table('alias_price_focus',alias.loc[alias.trade_date.between('2025-02-01','2025-03-07')])
    output.json('identifier_rules.json',{'identifier_alias_removals':aliases,
        'membership_replacements':config['d9']['membership_recovery'].get('membership_replacements',[]),
        'note':'Recorded project identifier rules do not prove price continuity; no series was spliced.'})
    return candidates,pd.DataFrame(coverage)


def load_weights_flags(access,config,start,end):
    weights,flags = [],[]
    for directory in [config['data']['processed_design_dir'],config['d9']['v2']['processed_oos_dir']]:
        for name,container in [('benchmark_weights',weights),('tradability_flags',flags)]:
            container.append(read_frame(access,Path(directory)/(name+'.parquet'),start,end))
    return pd.concat(weights,ignore_index=True),pd.concat(flags,ignore_index=True)


def run(root, args, output):
    config = yaml.safe_load((root/'config/config.yaml').read_text())
    report = preflight.Report()
    for fn,rest in [(preflight.check_repository,[json.loads((root/'config/d11_baseline.json').read_text())]),
                    (preflight.check_inputs,[config]),(preflight.check_registered_digests,[]),
                    (preflight.check_environment,[config]),(preflight.check_saved_runs,[])]:
        fn(report,*rest)
    output.json('preflight.json',{'stages':report.stages})
    if report.blocking: raise D12Error('Day3 preflight blocked; see preflight.json')
    source = root/'reports/d11/d11_v1'
    diagnostic = root/'reports/d11/d11_diagnostic_v1'
    source_identity = {p.name:verify_manifest(p) for p in [source,diagnostic]}
    verify_run(source)
    before = snapshot(root,config)
    original = json.loads((source/'inputs_before.json').read_text())
    parity = source_input_parity(before,original)
    output.json('inputs_before.json',before)
    output.json('source_identity.json',{'repository':report.stages[0]['detail'],
        'saved_runs':source_identity,'input_parity':parity,'source_run':'d11_v1',
        'repaired_r0':False,'historical_decision':'No-Go'})
    access = ResearchDataAccess(root)
    if access.mode != 'oos': raise D12Error('Original OOS marker required; no automatic unseal')
    start,end = pd.Timestamp(config['windows']['design_start']),pd.Timestamp(config['windows']['oos_end'])
    calendar = load_open_calendar(access,config,end_date=end)
    weights,flags = load_weights_flags(access,config,start,end)
    saved = {n:read_saved(source/(n+'.parquet')) for n in SAVED}
    timings = {}
    begin = time.perf_counter()
    candidates, coverage = scan_prices(access,config,weights,calendar,output,args.chunk_size)
    timings['p0_price_seconds'] = time.perf_counter()-begin
    print('D12 rebuilding original market and actual inventory/cash (no model fitting)',flush=True)
    begin = time.perf_counter()
    market,_ = prepare_oos_market(access,config,weights,flags)
    path = reconstruct_d11(saved['daily'],saved['trades'],market,float(config['d8']['main_capital_cny']),CostModel.from_config(config))
    output.parquet('reconstructed_accounting.parquet',path.daily)
    output.parquet('actual_daily_holdings.parquet',path.ending_weights.rename_axis(columns='instrument').stack().rename('weight').reset_index())
    timings['account_reconstruction_seconds'] = time.perf_counter()-begin
    important = candidates.loc[candidates.in_project_universe | candidates.instrument.isin(['SZ300114','SZ302132'])]
    propagated = price_propagation(important,weights,path,saved['predictions'],saved['labels_for_training_and_evaluation'],saved['orders'],saved['factor_exposures'])
    output.parquet('price_propagation.parquet',propagated)
    output.table('price_propagation_focus',propagated.loc[propagated.instrument.isin(['SZ300114','SZ302132'])] if not propagated.empty else propagated)
    provider = (root/Path(config['data']['qlib_provider_uri']).expanduser()).resolve()
    intervals = pd.read_csv(provider/'instruments/csi500.txt',sep=r'\s+',header=None,names=['instrument','start_date','end_date'])
    intervals['instrument'] = intervals.instrument.str.upper()
    for c in ['start_date','end_date']: intervals[c] = pd.to_datetime(intervals[c])
    output.table('provider_membership_intervals',intervals)
    snap = access.read_parquet(Path(config['data']['tushare_raw_dir'])/'index_weight_000905_SH.parquet',date_col='trade_date',start_date=start,end_date=end)
    snap['trade_date'] = parse_date_series(snap.trade_date)
    timing = membership_timing(weights,intervals,snap,calendar)
    output.table('membership_timeline',timing)
    proxy = suspected_membership_windows(weights,path,calendar)
    output.table('membership_proxy_windows',proxy)
    membership_status = 'blocked_no_official_full_set_evidence'
    if args.membership_evidence:
        evidence_path = Path(args.membership_evidence).resolve()
        if not evidence_path.is_relative_to(root/'data') or evidence_path.is_symlink():
            raise D12Error('Place optional independent membership evidence under data/ so it is covered by input snapshots')
        evidence = pd.read_csv(evidence_path,sep='\t')
        compared = official_membership_comparison(evidence,weights,path)
        output.table('official_membership_comparison',compared)
        membership_status = 'evidence_compared_pending_review'
    begin = time.perf_counter()
    estimates = rolling_estimates(market,ImpactEstimator.from_config(config))
    detail,groups,by_date,decisions = trade_groups(saved['trades'],saved['targets'],saved['predictions'],saved['audits'],estimates,market)
    decisions['target_active_at_cap'] = (~decisions.initial_construction_exempt.eq(True) &
        ((decisions.target_active_turnover-float(config['d7']['single_sided_turnover_limit'])).abs() <= float(config['d7']['constraint_tolerance'])))
    output.json('p1_summary.json',{'order_rows':len(detail),'filled_rows':int(detail.filled.sum()),
        'regular_decisions':int((~decisions.initial_construction_exempt.eq(True)).sum()),
        'regular_decisions_at_turnover_cap':int(decisions.target_active_at_cap.sum()),
        'cash_scaled_buy_rows':int(detail.cash_scaled.sum()),
        'ex_ante_above_5pct_rows':int(detail.ex_ante_above_limit.sum()),
        'actual_capacity_limited_rows':int(detail.participation_limited.sum()),
        'amount_estimate_missing_rows':int(detail.a_est_missing.sum()),
        'costs_reconciled':True,
        'expected_contribution_note':'signed requested trade weight times expected excess; no-cost, no-fill assumption; not realized profit',
        'capacity_shortfall_note':'diagnostic eligible-order cap shortfall before buy cash scaling; not a new order constraint'})
    output.parquet('p1_trade_detail.parquet',detail)
    for name,f in [('p1_trade_groups',groups),('p1_participation_by_decision_side',by_date),('p1_decisions',decisions)]: output.table(name,f)
    timings['p1_seconds'] = time.perf_counter()-begin
    replay = {'status':'not_requested','passed':False}
    if args.replay:
        begin = time.perf_counter()
        replay = full_replay(market,saved,flags,calendar,config,output)
        timings['r0_replay_seconds'] = time.perf_counter()-begin
    # Keep the registry immutable. Produce a dated evidence-linked proposal after P1.
    output.json('r1_registration_review.json',{'status':'frozen_spec_retained_execution_blocked',
        'frozen_spec_sha256':sha256(root/'config/d12_experiments.yaml'),
        'registry_sha256':sha256(root/'reports/d12/experiment_registry.tsv'),
        'p1_completed':True,'r0_replay':replay,
        'decision_k':.5,'execution_k_nominal':.5,'execution_k_stress':1.,
        'new_participation_constraint':False,'formal_r1_ready':False,
        'r1_zero_coefficient_replay_still_required':True})
    gate = {'formal_r1_ready':False,'repaired_r0_established':False,
        'p0_price_status':'candidates_and_coverage_pending_review',
        'p0_membership_status':membership_status,
        'why_blocked':['P0-12/P0-15/P0-17 are not automatically cleared by screens or parity',
                       'independent price/factor and official membership evidence require classification',
                       'A-type corrections require new inputs, dependency invalidation and a separate repaired R0',
                       'day4 R1 implementation needs its own full zero-coefficient replay'],
        'observed_price_candidates':len(candidates), 'in_universe_candidates':int(candidates.in_project_universe.sum()),
        'direct_holding_exposure_rows':int(propagated.beginning_actual_weight.gt(0).sum()) if not propagated.empty else 0,
        'reference_close_coverage':float(coverage.reference_close_rows.sum()/max(1,coverage.project_universe_rows.sum())),
        'reference_factor_ratio_coverage':float(coverage.reference_factor_ratio_rows.sum()/max(1,coverage.project_universe_rows.sum())),
        'timings':timings,'historical_decision':'No-Go','new_oos_validation':False}
    output.json('p0_stage_conclusion.json',gate)
    output.text('day3_report.md',f'''# D12 第三天：本机阶段证据

原结果参照：d11_v1；未建立修复 R0。历史 No-Go 不变。

- 全期价格候选：{len(candidates)} 行；处于项目股票池：{gate['in_universe_candidates']} 行。
- 独立原始收盘价覆盖率：{gate['reference_close_coverage']:.2%}；独立因子相邻变化覆盖率：{gate['reference_factor_ratio_coverage']:.2%}。
- 实际持仓/现金账本重建已通过；原 R0 全期重放通过：{replay.get('passed',False)}。
- P1 分组四项费用已与全部成交对账；初始建仓、主动交易、强制退出单列。
- 成分证据状态：{membership_status}。proxy 表是后见篮子对照，不是官方名单或权重。
- 正式 R1 仍阻断；新 R1 路径的零系数全期门尚未执行。

请先审阅 price_propagation_focus.tsv、alias_price_focus.tsv、price_candidates_top.tsv、price_coverage.tsv、membership_timeline.tsv 与 membership_proxy_windows.tsv。缺价与零涨跌停占位不等于无涨跌幅限制，复权异常候选也不等于已经确认的数据错误。P1 的 expected_trade_contribution 是事前信号量，不能当作已实现盈利。

只在独立证据定位根因后另建修复输入：价格问题需核查特征/标签/校准/权重/风险依赖；成分问题需核查生效日与公告可得时间，并补足合法权重来源。不得拼接别名价格、截断收益或倒推权重来缩小桥接。修复后重新决定预测缓存可否使用，不覆盖原 D11。
''')
    after = snapshot(root,config)
    assert_snapshot_unchanged(before,after)
    after_identity = {p.name:verify_manifest(p) for p in [source,diagnostic]}
    if source_identity != after_identity: raise D12Error('Source manifest identity changed during run')
    output.json('inputs_after.json',after)
    return {'status':'completed','inputs_unchanged':True,'formal_r1_ready':False,
            'source_identity':source_identity,'r0_replay':replay,'timings':timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True,help='new immutable day3 run ID')
    parser.add_argument('--replay',action='store_true',help='also reproduce original R0 over the full saved window')
    parser.add_argument('--chunk-size',type=int,default=80,help='price loading memory bound; does not change scope')
    parser.add_argument('--membership-evidence',help='optional full official membership sets under data/, TSV')
    args = parser.parse_args()
    if args.chunk_size<1: parser.error('--chunk-size must be positive')
    output = Day3Output(ROOT,args.run_id)
    started = time.perf_counter()
    try:
        manifest = run(ROOT,args,output)
    except Exception as error:
        manifest = {'status':'failed','inputs_unchanged':False,'formal_r1_ready':False,
                    'error':f'{type(error).__name__}: {error}'}
        output.text('failure.txt',traceback.format_exc())
        print(manifest['error'],file=sys.stderr,flush=True)
    manifest.update(run_id=args.run_id,recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    elapsed_seconds=time.perf_counter()-started,output_sha256=output.hashes())
    output.json('manifest.json',manifest)
    print(f'Wrote {output.path}; status={manifest["status"]}; formal_r1_ready=False',flush=True)
    return 0 if manifest['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

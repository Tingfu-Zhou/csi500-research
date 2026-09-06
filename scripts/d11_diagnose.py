"""Diagnose saved D11 returns, IC, execution, attribution and dynamic impact2x."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series
from src.d3_core import CostModel
from src.d6_core import compute_raw_style_history, tushare_to_qlib
from src.d10_core import DiagnosticError, require_close, unique_frame
from src.d11_core import simulate_causal
from src.d11_io import D11Output, sha256, verify_repository, snapshot_inputs, assert_snapshot_unchanged
from src.d11_diagnostics import (COST_COLUMNS, attribute_d11, compare_common_dates,
    execution_diagnostics, performance_and_costs, reconstruct_d11, signal_diagnostics)
from scripts.d11_pipeline import verify_run, read_frame, prepare_oos_market, load_open_calendar
from scripts.d6_pipeline import assemble_exposures
from src.d9_core import summarize_monthly_performance

REQUIRED = ['daily.parquet','monthly.parquet','trades.parquet','orders.parquet','targets.parquet',
            'audits.parquet','predictions.parquet','labels_for_training_and_evaluation.parquet',
            'factor_exposures.parquet','summary.json','manifest.json','inputs_before.json']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    """Represent genuinely undefined small-sample statistics as JSON null."""
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp): return str(value)
    return value


def run_path(root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise DiagnosticError('Use a simple run ID; no path separators')
    path = Path(root).resolve()/'reports/d11'/run_id
    if path.resolve()!=path: raise DiagnosticError('Run directory may not traverse symlinks')
    return path


def check_source(path):
    missing = [n for n in REQUIRED if not (path/n).is_file()]
    if missing:
        raise DiagnosticError('Missing existing D11 outputs: '+', '.join(missing)+
            '. Run this on the original computer containing d11_v1; do not retrain or delete that run.')
    return verify_run(path)


def preflight(root, config, source_id):
    source = run_path(root,source_id)
    checked = verify_repository(root,read_json(root/'config/d11_baseline.json'))
    verified = check_source(source)
    original = read_json(source/'inputs_before.json')
    for name in ['config/config.yaml','src/d11_core.py','scripts/d11_pipeline.py']:
        if sha256(root/name)!=original[name]['sha256']:
            raise DiagnosticError('Source-run configuration/causal simulator changed: '+name)
    missing = [str(p) for p in [root/'reports/d9_v2_main_daily.parquet',
        root/config['d6']['industry_mapping_file'],root/config['d6']['industry_catalog_file']]
        if not p.is_file()]
    if missing: raise DiagnosticError('Missing existing diagnostic inputs: '+', '.join(missing))
    if ResearchDataAccess(root).mode!='oos': raise DiagnosticError('Existing OOS marker required; no automatic unseal')
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','pyarrow','pyqlib','cvxpy','clarabel']}
    expected = read_json(source/'manifest.json')['preflight']['versions']
    for p in ['pyqlib','cvxpy','clarabel']:
        if versions[p]!=expected[p]: raise DiagnosticError(f'Use source-run {p}={expected[p]}; observed {versions[p]}')
    return {'repository':checked, 'source_run':source_id, 'source_verified':verified,
            'versions':versions,'preflight_passed':True,'new_oos_validation':False}


def read_saved(path):
    f = pd.read_parquet(path)
    for col in ['trade_date','sample_date','signal_date','execution_date','label_end_timestamp',
                'label_start_timestamp','model_cutoff','calibration_cutoff','month_end']:
        if col in f: f[col]=pd.to_datetime(f[col])
    return f


def load_inputs(access, config, saved_exposures, output):
    """Load original local data; rebuild DAILY exposures without fitting signals."""
    start, end = pd.Timestamp(config['windows']['oos_start']),pd.Timestamp(config['windows']['oos_end'])
    calendar = load_open_calendar(access,config,end_date=end)
    prior = calendar[calendar<start].max()
    weights, flags = [],[]
    for directory,lo,hi in [(config['data']['processed_design_dir'],prior,config['windows']['design_end']),
                            (config['d9']['v2']['processed_oos_dir'],start,end)]:
        weights.append(read_frame(access,Path(directory)/'benchmark_weights.parquet',lo,hi))
        flags.append(read_frame(access,Path(directory)/'tradability_flags.parquet',lo,hi))
    weights,flags = pd.concat(weights,ignore_index=True),pd.concat(flags,ignore_index=True)
    print('D11 diagnostic: loading original market and rebuilding daily factor exposures',flush=True)
    market,_ = prepare_oos_market(access,config,weights,flags)
    history = start-pd.Timedelta(days=int(config['d6']['history_buffer_calendar_days']))
    official = read_frame(access,Path(config['data']['tushare_raw_dir'])/'index_daily_H00905_CSI.parquet',history,end)
    unique_frame(official,['trade_date'],'official index')
    official = official.sort_values('trade_date').reset_index(drop=True)
    official['benchmark_return'] = official.close.pct_change(fill_method=None)
    frames=[]
    for year in range(history.year,end.year+1):
        f=read_frame(access,Path(config['data']['tushare_raw_dir'])/f'daily_basic_{year}.parquet',history,end)
        if 'instrument' not in f: f['instrument']=f.ts_code.map(tushare_to_qlib)
        frames.append(f.loc[f.instrument.isin(weights.instrument.unique())])
    basic=pd.concat(frames,ignore_index=True)
    d6=config['d6']
    raw=compute_raw_style_history(market[['trade_date','instrument','close']],official,basic,
        beta_window=int(d6['beta_window_trade_days']),beta_minimum_observations=int(d6['beta_minimum_observations']),
        momentum_window=int(d6['momentum_window_trade_days']),momentum_skip=int(d6['momentum_skip_trade_days']),
        liquidity_window=int(d6['liquidity_window_trade_days']),liquidity_minimum_observations=int(d6['liquidity_minimum_observations']))
    exposure,imputation,industries,styles=assemble_exposures(weights,raw,
        pd.read_parquet(access.root/d6['industry_mapping_file']),pd.read_parquet(access.root/d6['industry_catalog_file']),
        pd.DatetimeIndex(weights.trade_date.unique()).sort_values(),winsor_mad_multiple=float(d6['winsor_mad_multiple']))
    keys=['trade_date','instrument']
    old=saved_exposures.set_index(keys).sort_index()
    new=exposure.loc[exposure.trade_date.isin(saved_exposures.trade_date.unique())].set_index(keys).sort_index()
    if not old.index.equals(new.index): raise DiagnosticError('Rebuilt daily / saved signal exposure keys disagree')
    error=require_close(new[industries+styles+['market_cap','benchmark_weight']],
                        old[industries+styles+['market_cap','benchmark_weight']],
                        'daily/signal exposure parity',atol=1e-8,rtol=1e-10)
    output.json('exposure_parity.json',{'passed':True,'maximum_error':error,'signal_rows':len(old)})
    output.parquet('daily_exposures.parquet',exposure)
    output.parquet('daily_exposure_imputation.parquet',imputation)
    return market,exposure,flags,calendar,official,industries,styles


def check_parity(left,right,keys,columns,label,*,atol=1e-9):
    unique_frame(left,keys,label);unique_frame(right,keys,label)
    a,b=left.set_index(keys).sort_index(),right.set_index(keys).sort_index()
    if not a.index.equals(b.index): raise DiagnosticError(label+': keys differ')
    return require_close(a[columns],b[columns],label,atol=atol,rtol=2e-9)


def replay_stress(market,predictions,exposures,flags,calendar,config,industries,styles,saved,output):
    """First reproduce the baseline, then double impact with endogenous holdings."""
    results={}
    constrained=['style_'+s for s in config['d7']['constrained_style_factors']]
    if not set(constrained).issubset(styles): raise DiagnosticError('Missing constrained factors')
    for name,multiplier in [('baseline_replay',1.),('impact2x',2.)]:
        changed=deepcopy(config)
        changed['costs']['impact']['coefficient']*=multiplier
        count=0
        def progress(target,orders,audit):
            nonlocal count
            count+=1
            if count==1 or count%25==0: print(f'D11 {name}: decision {count} / {len(saved["audits"])}',flush=True)
        result=simulate_causal(market,predictions,exposures,flags,calendar,changed,industries,constrained,progress=progress)
        daily=result.nav.merge(saved['daily'][['trade_date','benchmark_return']],on='trade_date',how='left',validate='one_to_one')
        if not daily.trade_date.equals(saved['daily'].trade_date): raise DiagnosticError('Replay date mismatch')
        daily['portfolio_return']=daily.nav/daily.nav.shift(1,fill_value=float(config['d8']['main_capital_cny']))-1
        if name=='baseline_replay':
            errors={
                'daily':check_parity(daily,saved['daily'],['trade_date'],['nav','cash','transaction_cost'],'baseline account',atol=1e-4),
                'returns':check_parity(daily,saved['daily'],['trade_date'],['single_sided_turnover','portfolio_return'],'baseline returns'),
                'targets':check_parity(result.targets,saved['targets'],['signal_date','instrument'],['target_weight','pretrade_weight'],'baseline targets'),
                'orders':check_parity(result.orders,saved['orders'],['signal_date','instrument'],['order_units','signal_nav'],'baseline orders',atol=1e-5),
                'trades':check_parity(result.trades,saved['trades'],['execution_date','instrument'],['executed_units','total_cost'],'baseline fills',atol=1e-5)}
            output.json('baseline_replay_parity.json',{'passed':True,'maximum_errors':errors})
        else:
            reconstructed=reconstruct_d11(daily,result.trades,market,float(config['d8']['main_capital_cny']),CostModel.from_config(changed))
            output.parquet('impact2x_accounting.parquet',reconstructed.daily)
            for key in ['targets','orders','trades','audits']: output.parquet('impact2x_'+key+'.parquet',getattr(result,key))
            output.parquet('impact2x_daily.parquet',daily)
        p,m=summarize_monthly_performance(daily,newey_west_lags=config['d9']['newey_west_lags'])
        output.parquet(name+'_monthly.parquet',m)
        results[name]=p
    results['method']='fixed saved predictions; impact coefficient x2; causal reoptimization with actual cash/holdings; no model fitting'
    return results


def reference_gates(performance,attribution,stress,config):
    t=config['d9']['thresholds']
    share=attribution['specific_share_net']
    return {'annualized_excess_return':performance['annualized_excess_return']>=t['annualized_excess_return_min'],
        'information_ratio':performance['information_ratio']>=t['information_ratio_min'],
        'newey_west_t':performance['newey_west_t_minimum']>=t['newey_west_t_min'],
        'maximum_relative_drawdown':performance['maximum_relative_drawdown']<=t['maximum_relative_drawdown_max'],
        'specific_share':share>=t['specific_contribution_share_min'] if share is not None else None,
        'impact2x':stress['impact2x']['annualized_excess_return']>t['doubled_impact_annualized_excess_min']}


def run_diagnostics(root,config,source,output):
    saved={n:read_saved(source/(n+'.parquet')) for n in ['daily','monthly','trades','orders','targets','audits','predictions',
        'labels_for_training_and_evaluation','factor_exposures']}
    capital=float(config['d8']['main_capital_cny']);lags=config['d9']['newey_west_lags']
    perf,monthly,yearly=performance_and_costs(saved['daily'],capital,lags)
    original=read_json(source/'summary.json')['performance']
    for key in ['annualized_excess_return','information_ratio','newey_west_t_minimum','maximum_relative_drawdown']:
        require_close([perf['net'][key]],[original[key]],'source summary '+key)
    check_parity(monthly,saved['monthly'],['month'],['portfolio_return','benchmark_return','excess_return'],'saved monthly')
    output.parquet('monthly_performance.parquet',monthly);output.parquet('yearly_performance.parquet',yearly)
    ic,ic_year=signal_diagnostics(saved['predictions'],saved['labels_for_training_and_evaluation'],config['windows']['oos_end'])
    output.parquet('signal_ic.parquet',ic);output.parquet('yearly_ic.parquet',ic_year)
    output.text('signal_ic.tsv',ic.to_csv(sep='\t',index=False))
    # Small UTF-8 TSVs plus JSON are intentionally commit-friendly under existing ignore rules.
    for name,frame in [('yearly_performance',yearly),('yearly_ic',ic_year),('monthly_performance',monthly)]:
        output.text(name+'.tsv',frame.to_csv(sep='\t',index=False))
    access=ResearchDataAccess(root)
    inputs=load_inputs(access,config,saved['factor_exposures'],output)
    market,exposures,flags,calendar,official,industries,styles=inputs
    daily=saved['daily']
    require_close(daily.benchmark_return,official.set_index('trade_date').benchmark_return.reindex(daily.trade_date),'D11 official benchmark')
    old=read_frame(access,Path('reports/d9_v2_main_daily.parquet'),pd.Timestamp(config['windows']['oos_start']),config['windows']['oos_end'])
    if 'capital' in old:
        old=old.loc[old.capital.eq(capital)].copy()
    comparison,paired=compare_common_dates(old,daily,official,lags)
    output.parquet('common_monthly_comparison.parquet',paired)
    output.text('common_monthly_comparison.tsv',paired.to_csv(sep='\t',index=False))
    path=reconstruct_d11(daily,saved['trades'],market,capital,CostModel.from_config(config))
    output.parquet('reconstructed_accounting.parquet',path.daily)
    execution,exec_dates=execution_diagnostics(saved['trades'],saved['targets'],saved['audits'],path)
    output.parquet('execution_by_decision.parquet',exec_dates)
    output.text('execution_by_decision.tsv',exec_dates.to_csv(sep='\t',index=False))
    # Link each PnL term to the exact reported monthly excess (no simple sum approximation).
    ledger=daily[['trade_date','portfolio_return','benchmark_return']].merge(path.daily,on='trade_date',validate='one_to_one')
    ledger['benchmark_component']=-ledger.benchmark_return
    for name in COST_COLUMNS: ledger[name]=-ledger[name]/ledger.previous_nav
    columns=['holding_return','execution_return','benchmark_component']+COST_COLUMNS
    from src.d10_core import linked_monthly
    linked=linked_monthly(ledger,columns)
    require_close(linked[columns].sum(axis=1),linked.excess_return,'monthly PnL components')
    output.parquet('pnl_components_monthly.parquet',linked)
    output.text('pnl_components_monthly.tsv',linked.to_csv(sep='\t',index=False))
    print('D11 diagnostic: attributing actual holdings with lagged daily exposures',flush=True)
    ad,am,factors,fits,attribution=attribute_d11(daily,path,market,exposures,industries,styles)
    for name,frame in [('attribution_daily',ad),('attribution_monthly',am),('factor_returns',factors),('regression_audits',fits)]:
        output.parquet(name+'.parquet',frame)
    output.text('attribution_monthly.tsv',am.to_csv(sep='\t',index=False))
    require_close([attribution['annualized_contributions']['excess_return']],[perf['net']['annualized_excess_return']],'attribution total')
    stress=replay_stress(market,saved['predictions'],saved['factor_exposures'],flags,calendar,config,industries,styles,saved,output)
    summary={'classification':'post_oos_d11_return_and_execution_diagnostic','source_run':source.name,
        'historical_decision':'No-Go','new_oos_validation':False,'model_training_performed':False,
        'performance':perf,'yearly_performance':yearly.to_dict('records'),'yearly_ic':ic_year.to_dict('records'),
        'comparison':comparison,'execution':execution,'attribution':attribution,'stress':stress,
        'annualized_pnl_components':{k:float(linked[k].mean()*12) for k in columns},
        'accounting':{'passed':True,'maximum_nav_error_cny':float(path.daily.nav_error_cny.max()),
                      'maximum_daily_residual':float(path.daily.accounting_residual.abs().max())},
        'reference_gates':reference_gates(perf['net'],attribution,stress,config),
        'limitations':['post-hoc history; no fresh OOS','static SW2021 and source vintages not proven PIT',
            'fractional adjusted units, no round lots/dividend cash ledger','full-day VWAP/volume approximation',
            'unfilled marked opportunity is not executable profit','common dates do not isolate individual fixes',
            'cost add-back and factor residual are diagnostic definitions, not proven pure alpha']}
    return clean(summary)


def write_report(summary,output):
    s=summary;p=s['performance']['net'];a=s['attribution'];e=s['execution']
    fmt=lambda x: '未定义' if x is None else f'{x:.2%}'
    lines=['# D11 收益来源与交易损耗诊断','',f"来源运行：`{s['source_run']}`。本报告由本机明细计算；不训练模型，不改变历史 No-Go。",'',
        f"费后月频年化超额 **{p['annualized_excess_return']:.2%}**，IR **{p['information_ratio']:.3f}**，最低 NW-t **{p['newey_west_t_minimum']:.3f}**，最大相对回撤 **{p['maximum_relative_drawdown']:.2%}**。",'',
        '## 年度表现与预测能力','',
        '年度超额为该年区间组合累计收益减基准累计收益；最后一年不足全年，不冒充全年收益。IC 仅评价当时已预测且事后标签可用的股票，覆盖率单列。', '',
        '| 年度 | 月数 | 区间超额 | 相对回撤 | Rank IC | 标签评价覆盖率 |','|---|---:|---:|---:|---:|---:|']
    ic={r['year']:r for r in s['yearly_ic']}
    for r in s['yearly_performance']:
        i=ic[r['year']];value=i['rank_ic_mean']
        lines.append(f"| {r['year']} | {r['months']} | {fmt(r['period_excess_difference'])} | {fmt(r['maximum_relative_drawdown'])} | {value:.4f} | {fmt(i['evaluation_coverage'])} |" if value is not None else
                     f"| {r['year']} | {r['months']} | {fmt(r['period_excess_difference'])} | {fmt(r['maximum_relative_drawdown'])} | 未定义 | {fmt(i['evaluation_coverage'])} |")
    perf=s['performance']
    lines+=['',f"最长连续负超额月份：{perf['longest_negative_month_streak']}；最大回撤峰值：{perf['drawdown_peak']}，谷底：{perf['drawdown_trough']}，恢复：{perf['drawdown_recovery'] or '窗口内未恢复'}。",'',
        '## 费用与成交','',f"实际路径费用加回后的年化超额：{fmt(perf['same_path_gross']['annualized_excess_return'])}；与费后差额：{fmt(perf['cost_drag_annualized'])}。这不是无成本重跑。",'',
        '| 成本项目 | 人民币 |','|---|---:|']
    for k,v in e['costs_cny'].items(): lines.append(f'| {k} | {v:,.2f} |')
    lines+=['',f"订单记录 {e['order_records']:,}，发生非零成交的记录 {e['filled_records']:,}；按信号日价格计算的金额成交比例 {fmt(e['fill_fraction_at_signal_prices'])}。",'',
        f"常规目标主动单边换手均值 {fmt(e['regular_mean_target_active_turnover'])}；常规实际总单边换手均值 {fmt(e['regular_mean_executed_total_turnover'])}。二者定义不同，实际总换手包含强制退出。",'',
        '停牌/涨跌停、成交占比和现金限制的记录数可能重叠。未成交量乘当日 VWAP 到收盘价差，只是标记的机会收益，不能写成可实现的损失；价差也可能为负。', '',
        '## 收益归因','', '| 月频年化贡献 | 数值 |','|---|---:|']
    for k in ['industry','style','specific_gross','cost','specific_net','benchmark_bridge','unmodelled_bridge']:
        lines.append(f"| {k} | {fmt(a['annualized_contributions'][k])} |")
    lines+=['',f"费后特异性贡献占比 {fmt(a['specific_share_net'])}；三块归因残差检查通过：{a['three_block_residual_passed']}。",'',
        'specific_net 已包含 cost，不能将两者再次相加。行业、风格、费后特异性、官方指数桥接和未覆盖桥接共同闭合；残差不能直接称为纯 alpha。陈旧暴露贡献为重叠披露项。', '',
        '## 同日期对照与压力测试','',
        f"共同日期 {s['comparison']['common_start']} 至 {s['comparison']['common_end']}；相同官方基准下年化超额差 {fmt(s['comparison']['annualized_excess_delta'])}。",'',
        '对齐日期仍保留各自此前持仓历史，不能据此把增益分配给 E01/E04/E03。剔除日期及原基准差异见 summary.json。', '',
        f"冲击系数翻倍后的年化超额：{fmt(s['stress']['impact2x']['annualized_excess_return'])}。复用原预测，重算真实现金、成交及后续目标；此前先核对原系数重放与原账本一致。",'',
        '## 原门槛的事后对照','', '| 指标 | 达到原数值门槛 |','|---|---|']
    for k,v in s['reference_gates'].items(): lines.append(f"| {k} | {'未定义' if v is None else '是' if v else '否'} |")
    lines+=['','该表仅为事后数值对照，归因桥接质量另行报告，不是新的预注册 Go 验收。','',
        '## 下一轮研究判断','',
        '- 年度 IC 持续为正而费用占比较高：优先研究交易门槛、分数平滑和换手惩罚。',
        '- 年度 IC 变化明显：研究训练窗口和标签稳定性。',
        '- 训练表现与后续预测差距：需要另补训练期诊断，当前结果不能独立判定过拟合。',
        '- 先固定少量实验与评价标准，记录全部尝试，后续冻结版本做前向模拟。','',
        '## 证据与局限','',
        'manifest.json 记录源码、输入与输出哈希；诊断前后验证原运行及原始数据。生成的 Markdown/JSON/TSV 可直接 git add；详细 Parquet 保留本机。',
        *['- '+x for x in s['limitations']], '']
    output.text('d11_diagnostic_report.md','\n'.join(lines))


def verify_diagnostic(path):
    m=read_json(path/'manifest.json')
    if m.get('status')!='completed' or not m.get('inputs_unchanged'):
        raise DiagnosticError('Diagnostic did not complete')
    for name,digest in m['output_sha256'].items():
        if Path(name).name!=name or sha256(path/name)!=digest: raise DiagnosticError('Diagnostic output hash mismatch: '+name)
    s=read_json(path/'summary.json')
    if not s['accounting']['passed'] or s['new_oos_validation'] or s['historical_decision']!='No-Go':
        raise DiagnosticError('Invalid diagnostic scope/accounting')
    return {'verified':True,'source_run':s['source_run'],'historical_decision':'No-Go',
            'reference_gates':s['reference_gates'],'scope':'diagnostic output hashes and completion; not raw-data replication'}


def run_once(root,config,source_id,run_id):
    root=Path(root).resolve();source=run_path(root,source_id);dest=run_path(root,run_id)
    if source==dest or dest.exists(): raise FileExistsError(f'{dest}: choose a new diagnostic ID; keep existing runs')
    checked=preflight(root,config,source_id)
    print('D11 diagnostic: hashing original inputs (no model fitting)',flush=True)
    before=snapshot_inputs(root,config)
    original=read_json(source/'inputs_before.json')
    for name,info in original.items():
        if (name.startswith(('data/','qlib/','models/')) or name=='reports/d9_v2_main_daily.parquet') and before.get(name)!=info:
            raise DiagnosticError('Raw input differs from source run: '+name)
    source_manifest_hash=sha256(source/'manifest.json')
    output=D11Output(root,run_id)
    output.json('inputs_before.json',before)
    output.json('started.json',checked)
    manifest={'status':'running','source_run':source_id,'source_manifest_sha256':source_manifest_hash,
        'preflight':checked,'inputs_unchanged':False}
    try:
        summary=run_diagnostics(root,config,source,output)
        after=snapshot_inputs(root,config)
        assert_snapshot_unchanged(before,after)
        check_source(source)
        if sha256(source/'manifest.json')!=source_manifest_hash: raise DiagnosticError('Source manifest changed')
        verify_repository(root,read_json(root/'config/d11_baseline.json'))
        output.json('inputs_after.json',after)
        output.json('summary.json',summary)
        write_report(summary,output)
        manifest.update(status='completed',inputs_unchanged=True)
    except Exception as exc:
        manifest.update(status='failed',error=str(exc))
        output.text('failure.txt',traceback.format_exc())
        output.json('manifest.json',manifest)
        raise
    manifest['output_sha256']={p.name:sha256(p) for p in output.path.iterdir() if p.is_file()}
    output.json('manifest.json',manifest)
    return output.path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',default='d11_v1')
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight',action='store_true')
    action.add_argument('--run-id')
    action.add_argument('--verify-run')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/'config/config.yaml').read_text())
    if args.verify_run: result=verify_diagnostic(run_path(ROOT,args.verify_run))
    elif args.preflight: result=preflight(ROOT,config,args.source_run)
    else: result=verify_diagnostic(run_once(ROOT,config,args.source_run,args.run_id))
    print(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()

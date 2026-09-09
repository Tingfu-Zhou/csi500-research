"""Day3: two-layer price evidence, real holdings, unchanged replay, and failure gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.d12_day3 import (price_audit, reference_crosscheck, price_propagation,
    membership_timing, suspected_membership_windows, official_membership_comparison, trade_groups)
from src.d12_day3_io import Day3Output, snapshot, source_input_parity, verify_manifest, full_replay
from src.d12_repo import D12Error
from src.d11_io import assert_snapshot_unchanged
from src.d10_core import DiagnosticError
from src.d11_core import simulate_causal
from src.d12_core import ImpactEstimator, rolling_estimates


def price_fixture():
    dates = pd.bdate_range('2025-02-19',periods=5)
    p = pd.DataFrame({'trade_date':dates,'instrument':'SZ302132','close':[10.,10.,np.nan,60.,60.],
                      'factor':[1.,1.,1.,6.,6.], 'high':[10.,10.,np.nan,60.,60.], 'low':[10.,10.,np.nan,60.,60.]})
    limits = p[['trade_date','instrument']].assign(up_limit=12.,down_limit=8.)
    return p,limits,dates


def test_raw_price_can_pass_limits_while_adjusted_factor_jump_is_flagged():
    p,lim,dates = price_fixture()
    audit = price_audit(p,lim,dates)
    row = audit.loc[audit.trade_date.eq(dates[3])].iloc[0]
    assert row.raw_close == 10
    assert not row.outside_raw_limits
    assert row.adjusted_jump_candidate and row.factor_jump_candidate
    assert row.mark_return == 5
    assert row.return_span_sessions == 2
    assert row.previous_mark_source_date == dates[1]


def test_missing_rows_do_not_compress_exchange_calendar_and_future_does_not_change_past():
    p,lim,dates = price_fixture()
    absent = p.drop(index=2)
    a = price_audit(absent,lim,dates)
    b = price_audit(p.assign(close=[10,10,np.nan,6000,9000]).drop(index=2),lim,dates)
    pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])
    assert a.iloc[3].return_span_sessions == 2
    assert a.iloc[2].stale_mark_sessions == 1


def test_limit_missing_zero_invalid_separate_and_not_unlimited():
    p,lim,dates = price_fixture()
    lim.loc[0,['up_limit','down_limit']] = np.nan
    lim.loc[1,['up_limit','down_limit']] = 0
    lim.loc[2,['up_limit','down_limit']] = [1,2]
    a = price_audit(p,lim,dates)
    assert list(a.limit_status[:3]) == ['missing','zero_placeholder','invalid']
    assert 'unlimited' not in set(a.limit_status)


def test_raw_high_outside_limits_detected_with_normal_close():
    p,lim,dates = price_fixture()
    p.loc[0,'high'] = 13.
    a = price_audit(p,lim,dates)
    assert a.iloc[0].outside_raw_limits


def test_independent_factors_use_relative_changes_not_absolute_scale():
    p,lim,dates = price_fixture()
    a = price_audit(p,lim,dates)
    reference = p[['trade_date','instrument']].assign(reference_close=10.,reference_factor=100.)
    result = reference_crosscheck(a,reference)
    assert result.iloc[1].factor_reference_error == 0
    assert result.iloc[3].factor_reference_error == 5
    assert result.iloc[3].reference_mismatch
    reference.loc[3:,'reference_factor'] = 600.
    assert reference_crosscheck(a,reference).factor_reference_error.fillna(0).eq(0).all()


def test_duplicate_prices_or_limits_stop_the_audit():
    p,lim,dates = price_fixture()
    with pytest.raises(DiagnosticError): price_audit(pd.concat([p,p.iloc[:1]]),lim,dates)
    with pytest.raises(D12Error): price_audit(p,pd.concat([lim,lim.iloc[:1]]),dates)


def test_no_actual_holding_does_not_classify_anomaly_as_b_only():
    p,lim,dates = price_fixture()
    audit = price_audit(p,lim,dates).iloc[[3]]
    weights = p[['trade_date','instrument']].assign(weight=.2)
    zeros = pd.DataFrame(0.,index=dates,columns=['SZ302132'])
    path = SimpleNamespace(beginning_weights=zeros,ending_weights=zeros)
    pred = pd.DataFrame({'sample_date':[dates[3]],'instrument':['SZ302132']})
    labels = pred.assign(label_start_timestamp=dates[1],label_end_timestamp=dates[4])
    orders = pd.DataFrame(columns=['instrument','signal_date','execution_date'])
    ex = p[['trade_date','instrument']]
    result = price_propagation(audit,weights,path,pred,labels,orders,ex).iloc[0]
    assert result.beginning_actual_weight == 0
    assert result.label_windows_crossing_date == 1
    assert result.classification == 'A_possible_pending_evidence'


def membership_fixture():
    dates = pd.bdate_range('2026-06-01','2026-07-02')
    weights = pd.DataFrame([(d,n,.5) for d in dates for n in (['A','B'] if d<pd.Timestamp('2026-07-01') else ['A','C'])],columns=['trade_date','instrument','weight'])
    holdings = pd.DataFrame({'A':.5,'B':.4,'C':0},index=dates)
    path = SimpleNamespace(ending_weights=holdings)
    intervals = pd.DataFrame([('A',dates[0],dates[-1]),('B',dates[0],pd.Timestamp('2026-06-30')),('C',pd.Timestamp('2026-07-01'),dates[-1])],columns=['instrument','start_date','end_date'])
    return dates,weights,path,intervals


def test_provider_and_project_switch_dates_are_observed_not_called_official():
    dates,w,path,intervals = membership_fixture()
    t = membership_timing(w,intervals,pd.DataFrame({'trade_date':[dates[0],pd.Timestamp('2026-07-01')]}),dates)
    row = t.loc[t.trade_date.eq('2026-07-01')].iloc[0]
    assert row.processed_added == row.provider_added == 1
    assert row.session_in_month == 1
    assert not t.official_timing_confirmed.any()


def test_proxy_uses_real_holdings_including_unsold_removed_stock_not_targets():
    dates,w,path,_ = membership_fixture()
    result = suspected_membership_windows(w,path,dates)
    assert result.trade_date.min() == pd.Timestamp('2026-06-15')
    assert result.trade_date.max() == pd.Timestamp('2026-06-30')
    assert result.proxy_removed_actual_weight.eq(.4).all()
    assert result.official_added_weight.isna().all()
    assert result.classification.str.contains('not_official').all()


def test_official_list_comparison_counts_all_residual_holdings_and_availability():
    dates,w,path,_ = membership_fixture()
    evidence = pd.DataFrame({'effective_date':['2026-06-15']*2,'available_at':['2026-06-15T10:00:00+08:00']*2,
                             'instrument':['A','C'],'source_url':['https://example.test/official']*2})
    result = official_membership_comparison(evidence,w,path)
    assert result.actual_holdings_outside_official_weight.eq(.4).all()
    assert result.iloc[0].available_before_decision_day == False
    assert result.iloc[1].available_before_decision_day == True
    assert result.iloc[-1].removed_still_in_universe == 0
    assert result.iloc[-1].actual_holdings_outside_official_weight == .4
    with pytest.raises(D12Error,match='full constituent'): official_membership_comparison(evidence.iloc[:1],w,path)


def replay_fixture():
    from test_d11_core import simulation_fixture
    market,pred,ex,flags,calendar,config = simulation_fixture()
    # The real saved D6/D11 frame carries these two descriptive strings next to
    # the one-hot columns.  Keeping the Chinese name here prevents a prefix scan
    # from regressing into the production failure seen in day3_v1.
    ex = ex.assign(industry_one=1., industry_factor='industry_one', industry_name='汽车')
    config['d7']['constrained_style_factors'] = []
    result = simulate_causal(market,pred,ex,flags,calendar,config,['industry_one'],[])
    daily = result.nav.copy()
    daily['portfolio_return'] = daily.nav/daily.nav.shift(1,fill_value=1000.)-1
    saved = {'daily':daily,'targets':result.targets,'orders':result.orders,'trades':result.trades,'audits':result.audits,
             'predictions':pred,'factor_exposures':ex}
    return config,calendar,market,pred,ex,flags,saved


def test_full_original_replay_passes_but_neither_r1_zero_gate_nor_p0_gate(tmp_path):
    c,cal,m,p,e,f,saved = replay_fixture()
    out = Day3Output(tmp_path,'replay')
    parity = full_replay(m,saved,f,cal,c,out)
    assert parity['passed'] and parity['full_period']
    assert parity['decisions'] == len(saved['audits'])
    assert not parity['r1_zero_coefficient_gate_satisfied']
    assert not parity['p0_quality_certified']


def test_replay_uses_numeric_industry_dummies_and_ignores_descriptive_strings(tmp_path):
    c,cal,m,p,e,f,saved = replay_fixture()
    assert saved['factor_exposures']['industry_name'].eq('汽车').all()
    assert saved['factor_exposures']['industry_factor'].eq('industry_one').all()
    parity = full_replay(m,saved,f,cal,c,Day3Output(tmp_path,'industry_schema'))
    assert parity['passed']


def test_replay_fails_for_changed_orders_even_when_daily_nav_is_unchanged(tmp_path):
    c,cal,m,p,e,f,saved = replay_fixture()
    saved['orders'].loc[0,'order_units'] += .1
    with pytest.raises(DiagnosticError): full_replay(m,saved,f,cal,c,Day3Output(tmp_path,'bad'))


def test_p1_costs_group_unfilled_initial_orders_and_future_participation():
    c,cal,m,p,e,f,saved = replay_fixture()
    # Score isn't needed by the simulator fixture; P1 needs it.
    p = p.assign(score_z=1.)
    estimates = rolling_estimates(m,ImpactEstimator.from_config(c))
    detail,groups,by_side,decisions = trade_groups(saved['trades'],saved['targets'],p,saved['audits'],estimates,m)
    assert groups.total_cost.sum() == pytest.approx(saved['trades'].total_cost.sum())
    assert detail.a_est_missing.all()  # fixture shorter than required 15 sessions
    assert detail.ex_ante_participation.isna().all()
    assert 'initial' in set(detail.trade_type)
    assert len(decisions) == len(saved['audits'])
    other = m.copy();other['daily_amount_cny'] = 100.
    perturbed = trade_groups(saved['trades'],saved['targets'],p,saved['audits'],estimates,other)[0]
    pd.testing.assert_series_equal(detail.ex_ante_participation,perturbed.ex_ante_participation)
    assert not detail.realized_participation.equals(perturbed.realized_participation)


def test_p1_forced_sell_has_no_invented_score_or_expected_return():
    c,cal,m,p,e,f,saved = replay_fixture()
    trade = saved['trades'].iloc[[0]].copy()
    trade['order_units'] = -1.;trade['executed_units'] = -.5;trade['unfilled_units'] = -.5
    trade['executed_trade_amount'] = -5.
    # Deliberately absent from targets and prediction cross-section.
    name = trade.instrument.iloc[0]
    target = saved['targets'].loc[~saved['targets'].instrument.eq(name)]
    pred = p.loc[~p.instrument.eq(name)].assign(score_z=1.)
    audit = saved['audits'].copy();audit['initial_construction_exempt'] = False
    estim = rolling_estimates(m,ImpactEstimator.from_config(c))
    detail,groups,_,_ = trade_groups(trade,target,pred,audit,estim,m)
    assert detail.iloc[0].trade_type == 'forced_exit'
    assert detail.iloc[0].score_strength == 'unavailable'
    assert pd.isna(detail.iloc[0].expected_trade_contribution)
    assert groups.total_cost.sum() == pytest.approx(trade.total_cost.sum())


def test_source_snapshot_detects_changed_data_not_new_d12_code():
    old = {'data/a':{'sha256':'a','bytes':1},'src/d11_core.py':{'sha256':'b','bytes':2}}
    new = {**old,'src/d12_day3.py':{'sha256':'c','bytes':3}}
    assert source_input_parity(new,old)['passed']
    new['data/a'] = {'sha256':'d','bytes':1}
    with pytest.raises(D12Error,match='identity changed'): source_input_parity(new,old)


def test_day3_snapshot_excludes_only_outputs_but_detects_data_changes(tmp_path):
    (tmp_path/'data').mkdir();(tmp_path/'data/input').write_text('original')
    c = {'data':{'qlib_provider_uri':str(tmp_path/'provider')}}
    before = snapshot(tmp_path,c)
    out = Day3Output(tmp_path,'a');out.json('result.json',{'a':1})
    assert_snapshot_unchanged(before,snapshot(tmp_path,c))
    (tmp_path/'data/input').write_text('changed')
    with pytest.raises(DiagnosticError): assert_snapshot_unchanged(before,snapshot(tmp_path,c))
    with pytest.raises(FileExistsError): Day3Output(tmp_path,'a')
    with pytest.raises(D12Error): Day3Output(tmp_path,'../d11')
    with pytest.raises(D12Error): out.json('../escape',{})


def test_source_manifest_tampering_is_detected(tmp_path):
    from src.d12_repo import sha256
    p = tmp_path/'a.json';p.write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'completed','inputs_unchanged':True,'output_sha256':{'a.json':sha256(p)}}))
    assert verify_manifest(tmp_path)['outputs_verified'] == 1
    p.write_text('{"changed":true}')
    with pytest.raises(D12Error): verify_manifest(tmp_path)


def test_no_independent_reference_or_limits_is_a_coverage_gap_not_a_false_pass():
    p,lim,dates = price_fixture()
    a = price_audit(p,lim.iloc[:0],dates)
    reference = p[['trade_date','instrument']].iloc[:0].assign(reference_close=pd.Series(dtype=float),reference_factor=pd.Series(dtype=float))
    result = reference_crosscheck(a,reference)
    assert result.limit_status.eq('missing').all()
    assert result.raw_reference_error.isna().all()
    assert result.reference_factor_ratio.isna().all()
    assert result.adjusted_jump_candidate.any()


def test_day3_orchestration_with_real_evidence_functions_and_real_replay(tmp_path, monkeypatch):
    """Stub only environment/provider access; exercise manifests, all stages and output gates."""
    import argparse
    import yaml
    from scripts import d12_day3 as cli
    from src.d12_repo import sha256
    c,cal,m,p,e,f,saved = replay_fixture()
    c['windows']['design_start'] = str(cal[0].date())
    c['data']['qlib_provider_uri'] = str(tmp_path/'provider')
    c['d9']['membership_recovery']['identifier_alias_removals'] = []
    c['d7']['constrained_style_factors'] = []
    (tmp_path/'config').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'config/config.yaml').write_text(yaml.safe_dump(c))
    (tmp_path/'config/d11_baseline.json').write_text('{}')
    (tmp_path/'config/d12_experiments.yaml').write_text('r1: frozen')
    (tmp_path/'reports/d12').mkdir(parents=True)
    (tmp_path/'reports/d12/experiment_registry.tsv').write_text('id\tstatus\nR1\tfrozen\n')
    (tmp_path/'data/input').write_text('original')
    weights = m[['trade_date','instrument']].assign(weight=.5)
    prices = m.assign(factor=1.,high=m.close,low=m.close)
    lim = m[['trade_date','instrument']].assign(up_limit=12.,down_limit=8.)
    raw = tmp_path/c['data']['tushare_raw_dir'];raw.mkdir(parents=True,exist_ok=True)
    weights[['trade_date']].drop_duplicates().to_parquet(raw/'index_weight_000905_SH.parquet')
    provider = tmp_path/'provider/instruments';provider.mkdir(parents=True)
    (provider/'csi500.txt').write_text('A\t2020-01-02\t2020-01-08\nB\t2020-01-02\t2020-01-08\n')
    saved['predictions'] = p.assign(score_z=1.)
    saved['daily']['benchmark_return'] = 0.
    saved['labels_for_training_and_evaluation'] = p.assign(label_start_timestamp=p.sample_date,
                                                           label_end_timestamp=p.sample_date+pd.Timedelta(days=7))
    source = tmp_path/'reports/d11/d11_v1';source.mkdir(parents=True)
    diagnostic = tmp_path/'reports/d11/d11_diagnostic_v1';diagnostic.mkdir()
    for name,frame in saved.items(): frame.to_parquet(source/(name+'.parquet'),index=False)
    original = snapshot(tmp_path,c)
    (source/'inputs_before.json').write_text(json.dumps(original))
    for directory in [source,diagnostic]:
        hashes = {p.name:sha256(p) for p in directory.iterdir()}
        (directory/'manifest.json').write_text(json.dumps({'status':'completed','inputs_unchanged':True,'output_sha256':hashes}))
    class Access:
        def __init__(self, root): self.root=root;self.mode='oos'
        def read_parquet(self,path,**kw): return pd.read_parquet(self.root/path)
    monkeypatch.setattr(cli,'ResearchDataAccess',Access)
    monkeypatch.setattr(cli,'load_open_calendar',lambda *a,**k:cal)
    monkeypatch.setattr(cli,'load_weights_flags',lambda *a:(weights,f))
    monkeypatch.setattr(cli,'prepare_oos_market',lambda *a:(m,pd.DataFrame()))
    monkeypatch.setattr(cli,'load_qlib_market',lambda access,batch,*a:prices.loc[prices.instrument.isin(batch)])
    monkeypatch.setattr(cli,'load_limits',lambda *a:(lim,[]))
    monkeypatch.setattr(cli,'verify_run',lambda *a:{'verified':True})
    def check(report,*args):
        with report.stage('synthetic_environment') as entry: entry['detail']={'fixture':True}
    for name in ['check_repository','check_inputs','check_registered_digests','check_environment','check_saved_runs']:
        monkeypatch.setattr(cli.preflight,name,check)
    out = Day3Output(tmp_path,'day3_fixture')
    result = cli.run(tmp_path,argparse.Namespace(chunk_size=1,replay=True,membership_evidence=None),out)
    assert result['status'] == 'completed' and result['inputs_unchanged']
    assert result['r0_replay']['passed']
    assert not result['formal_r1_ready']
    gate = json.loads((out.path/'p0_stage_conclusion.json').read_text())
    assert gate['reference_close_coverage'] == 0
    assert not gate['repaired_r0_established']
    assert (out.path/'p1_trade_groups.tsv').is_file()
    assert json.loads((out.path/'inputs_before.json').read_text()) == json.loads((out.path/'inputs_after.json').read_text())

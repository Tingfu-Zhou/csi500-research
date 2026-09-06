"""Pipeline integration uses synthetic data; Git and artifact guards tested separately."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import d11_pipeline as pipeline
from src.d11_io import D11Output, verify_repository, git, git_blob, snapshot_inputs, assert_snapshot_unchanged
from src.d10_core import DiagnosticError
from src.d11_core import CausalityError


def synthetic_data(root, monkeypatch):
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/config.yaml').read_text())
    config['windows']['oos_end'] = '2019-03-29'
    config['data']['qlib_provider_uri'] = str(root/'provider')
    config['d7'].update(stock_active_limit=.1,industry_active_limit=1.,style_active_limit=1.,single_sided_turnover_limit=1.,constrained_style_factors=[])
    config['d8']['main_capital_cny'] = 10000.
    config['d9']['newey_west_lags'] = [0,1]
    (root/'config').mkdir()
    (root/'config/config.yaml').write_text(yaml.safe_dump(config))
    dates = pd.bdate_range('2014-01-01','2019-03-29')
    symbols = [f'S{n:02}' for n in range(40)]
    weights = pd.DataFrame([(d,n,1/40) for d in dates for n in symbols], columns=['trade_date','instrument','weight'])
    flags = weights[['trade_date','instrument']].assign(is_st=False,is_listed_under_120d=False,is_below_amount_floor=False)
    for key, mask in [('design',weights.trade_date < '2019-01-01'),('oos_v2',weights.trade_date >= '2019-01-01')]:
        path = root/'data/processed'/key
        path.mkdir(parents=True)
        weights.loc[mask].to_parquet(path/'benchmark_weights.parquet',index=False)
        flags.loc[mask].to_parquet(path/'tradability_flags.parquet',index=False)
    (root/'data/.window_mode.json').write_text('{"mode":"oos"}')
    raw = root/'data/raw/tushare'
    raw.mkdir(parents=True)
    pd.DataFrame({'cal_date':dates,'is_open':1}).to_parquet(raw/'trade_cal.parquet',index=False)
    pd.DataFrame({'trade_date':dates,'close':1000*np.exp(np.arange(len(dates))*.00005)}).to_parquet(raw/'index_daily_H00905_CSI.parquet',index=False)
    pos = weights.instrument.map({n:i for i,n in enumerate(symbols)}).to_numpy()
    time = weights.trade_date.map({d:i for i,d in enumerate(dates)}).to_numpy()
    prices = weights[['trade_date','instrument']].assign(vwap=10*np.exp(time*(pos-20)*.000015))
    market = prices.assign(close=prices.vwap,daily_amount_cny=1e8,volatility=.02,exchange_can_buy=True,exchange_can_sell=True)
    def features(access, instruments, samples, config):
        frame = weights[['trade_date','instrument']].rename(columns={'trade_date':'sample_date'}).assign(f=pos.astype(float))
        return frame.loc[frame.sample_date.isin(samples)],pd.DataFrame({'feature':['f'],'expression':['synthetic']})
    def exposures(access,config,w,signals):
        e = w.loc[w.trade_date.isin(signals)].rename(columns={'weight':'benchmark_weight'})
        return e,pd.DataFrame({'passed':[True]}),pd.DataFrame({'passed':[True]}),[],[],pd.DataFrame()
    monkeypatch.setattr(pipeline,'load_weekly_alpha158',features)
    monkeypatch.setattr(pipeline,'load_vwap_by_year',lambda *a:prices)
    monkeypatch.setattr(pipeline,'build_oos_exposures',exposures)
    monkeypatch.setattr(pipeline,'prepare_oos_market',lambda *a:(market,pd.DataFrame()))
    monkeypatch.setattr(pipeline,'preflight',lambda *a:{'synthetic_fixture':True})
    return config


def test_complete_synthetic_pipeline_and_tamper_detection(tmp_path,monkeypatch):
    config = synthetic_data(tmp_path,monkeypatch)
    run = pipeline.run_once(tmp_path,config,{},'synthetic_v1')
    assert pipeline.verify_run(run)['verified']
    summary = json.loads((run/'summary.json').read_text())
    assert summary['historical_decision'] == 'No-Go'
    assert summary['prediction_only_dates_without_in_window_t1'] == ['2019-03-29']
    pred = pd.read_parquet(run/'predictions.parquet')
    assert pred.groupby('sample_date').size().eq(40).all()
    before = json.loads((run/'inputs_before.json').read_text())
    assert before == json.loads((run/'inputs_after.json').read_text())
    with pytest.raises(FileExistsError):
        pipeline.run_once(tmp_path,config,{},'synthetic_v1')
    with (run/'orders.parquet').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(CausalityError,match='hash mismatch'):
        pipeline.verify_run(run)


def test_failed_run_retains_manifest_and_forbids_reuse(tmp_path,monkeypatch):
    config = synthetic_data(tmp_path,monkeypatch)
    def fail(*args): raise CausalityError('intentional infeasible fixture')
    monkeypatch.setattr(pipeline,'run_research',fail)
    with pytest.raises(CausalityError,match='intentional'):
        pipeline.run_once(tmp_path,config,{},'failed_v1')
    manifest = json.loads((tmp_path/'reports/d11/failed_v1/manifest.json').read_text())
    assert manifest['status'] == 'failed' and not manifest['inputs_unchanged']
    with pytest.raises(CausalityError,match='did not complete'):
        pipeline.verify_run(tmp_path/'reports/d11/failed_v1')


def test_snapshot_detects_input_mutation_and_excludes_new_outputs(tmp_path):
    (tmp_path/'data').mkdir()
    (tmp_path/'data/source').write_text('original')
    config = {'data':{'qlib_provider_uri':str(tmp_path/'provider')}}
    before = snapshot_inputs(tmp_path,config)
    output = D11Output(tmp_path,'test_v1')
    output.json('result.json',{'test':True})
    assert_snapshot_unchanged(before,snapshot_inputs(tmp_path,config))
    (tmp_path/'data/source').write_text('mutated')
    with pytest.raises(DiagnosticError,match='Protected inputs changed'):
        assert_snapshot_unchanged(before,snapshot_inputs(tmp_path,config))


@pytest.mark.parametrize('run_id',['../outside','a/b','', 'x'*81])
def test_output_path_rejects_escape(tmp_path,run_id):
    with pytest.raises(DiagnosticError): D11Output(tmp_path,run_id)


def test_output_path_rejects_symlink(tmp_path):
    (tmp_path/'reports').mkdir()
    (tmp_path/'elsewhere').mkdir()
    (tmp_path/'reports/d11').symlink_to(tmp_path/'elsewhere',target_is_directory=True)
    with pytest.raises(DiagnosticError): D11Output(tmp_path,'v1')


def test_repository_guard_real_git_branch_tag_and_protected_files(tmp_path):
    def command(*args): return git(tmp_path,*args)
    command('init','-b','d11')
    command('config','user.name','D11 test')
    command('config','user.email','d11-test@example.invalid')
    (tmp_path/'历史.md').write_text('frozen evidence')
    command('add','.')
    command('commit','-m','fixture')
    command('tag','-a','frozen-oos','-m','fixture tag')
    baseline = {'base_commit':command('rev-parse','HEAD'),'frozen_commit':command('rev-parse','HEAD'),
                'frozen_tag_object':command('rev-parse','frozen-oos'),
                'protected_blobs':{'历史.md':git_blob(tmp_path/'历史.md')}}
    assert verify_repository(tmp_path,baseline)['branch'] == 'd11'
    command('switch','-c','main')
    with pytest.raises(DiagnosticError,match='requires d11'): verify_repository(tmp_path,baseline)
    command('switch','d11')
    (tmp_path/'历史.md').write_text('modified')
    with pytest.raises(DiagnosticError,match='tracked files differ'): verify_repository(tmp_path,baseline)
    (tmp_path/'历史.md').write_text('frozen evidence')
    (tmp_path/'src').mkdir()
    (tmp_path/'src/new.py').write_text('uncommitted')
    with pytest.raises(DiagnosticError,match='uncommitted'): verify_repository(tmp_path,baseline)

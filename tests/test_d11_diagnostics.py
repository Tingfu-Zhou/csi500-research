"""Economic identities, missing-data boundaries and real-solver diagnostic flow."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.d3_core import CostModel
from src.d10_core import DiagnosticError
from src.d11_diagnostics import (attribute_d11, compare_common_dates, d11_price_panels,
    performance_and_costs, reconstruct_d11, signal_diagnostics)
from src.d11_core import execute_orders, freeze_orders
from scripts import d11_diagnose as diagnostic


def ledger_fixture():
    signal,execution=pd.Timestamp('2020-01-02'),pd.Timestamp('2020-01-03')
    costs=CostModel(.001,0.,((pd.Timestamp('1900-01-01'),pd.Timestamp('2100-01-01'),0.),),.5,.05)
    orders=freeze_orders(pd.Series({'A':.6,'B':.4}),pd.Series(dtype=float),1000.,
                         pd.Series({'A':10.,'B':10.}),pd.Series({'A':True,'B':True}),signal,execution)
    day=pd.DataFrame({'instrument':['A','B'],'vwap':[11.,10.],'close':[12.,9.],
        'daily_amount_cny':[1e6,1e6],'volatility':[.02,.02],
        'exchange_can_buy':[True,False],'exchange_can_sell':[True,True]}).set_index('instrument')
    units,cash,trades=execute_orders(pd.Series(dtype=float),1000.,orders,day,execution,costs)
    initial=day.reset_index().assign(trade_date=signal,close=10.,vwap=10.)
    market=pd.concat([initial,day.reset_index().assign(trade_date=execution)],ignore_index=True)
    nav=cash+units.mul(day.close).sum()
    daily=pd.DataFrame({'trade_date':[signal,execution],'nav':[1000.,nav],'cash':[1000.,cash],
        'portfolio_return':[0.,nav/1000.-1],'benchmark_return':[0.,.01],
        'transaction_cost':[0.,trades.total_cost.sum()],
        'single_sided_turnover':[0.,trades.executed_trade_amount.abs().sum()/2000.]})
    return daily,trades,market,costs


def test_units_ledger_cash_fee_and_intraday_pnl_identity():
    daily,trades,market,costs=ledger_fixture()
    path=reconstruct_d11(daily,trades,market,1000.,costs)
    assert path.daily.accounting_residual.abs().max()<1e-12
    assert path.daily.execution_return.iloc[-1]==pytest.approx(.06)
    assert path.beginning_weights.iloc[-1].sum()==0
    assert trades.loc[trades.instrument.eq('B'),'executed_units'].iloc[0]==0
    assert path.orders.loc[path.orders.instrument.eq('B'),'unfilled_to_close_mark_cny'].iloc[0]==-40.


@pytest.mark.parametrize('column', ['executed_units','total_cost','signal_units'])
def test_ledger_detects_tampered_trade(column):
    daily,trades,market,costs=ledger_fixture()
    trades.loc[0,column]+=1.
    with pytest.raises(DiagnosticError): reconstruct_d11(daily,trades,market,1000.,costs)


def test_d11_marks_do_not_fallback_to_vwap_or_seed_warmup():
    dates=pd.bdate_range('2020-01-01',periods=4)
    m=pd.DataFrame({'trade_date':dates,'instrument':'A','close':[10.,np.nan,12.,np.nan],
                    'vwap':[10.,11.,np.nan,np.nan]})
    mark,vwap=d11_price_panels(m,dates[1])
    assert pd.isna(mark.loc[dates[1],'A'])
    assert mark.loc[dates[3],'A']==12.
    assert pd.isna(vwap.loc[dates[2],'A'])


def test_ic_excludes_immature_labels_without_removing_predictions():
    d=pd.Timestamp('2026-08-21');names=list('ABCDEFGHIJ')
    p=pd.DataFrame({'sample_date':d,'instrument':names,'prediction':np.arange(10.),'feature_available':True})
    labels=pd.DataFrame({'sample_date':d,'instrument':names,'excess_return':np.arange(10.),'label_end_timestamp':pd.Timestamp('2026-08-28')})
    labels.loc[9,'label_end_timestamp']=pd.Timestamp('2026-09-01')
    p.loc[8,'feature_available']=False
    by_date,by_year=signal_diagnostics(p,labels,'2026-08-31')
    assert by_date.predicted_rows.iloc[0]==10
    assert by_date.evaluated_rows.iloc[0]==8
    assert by_date.rank_ic.iloc[0]==pytest.approx(1.)
    assert by_year.evaluation_coverage.iloc[0]==.8
    assert len(p)==10


def return_fixture():
    dates=pd.bdate_range('2020-01-01','2020-04-30')
    rp=np.sin(np.arange(len(dates)))*.001+.0002
    return pd.DataFrame({'trade_date':dates,'portfolio_return':rp,'benchmark_return':.0001})


def test_common_dates_use_one_official_benchmark_and_disclose_exclusions():
    new=return_fixture();old=new.iloc[1:].copy();old.loc[old.index[0],'benchmark_return']=0.
    official=new[['trade_date','benchmark_return']]
    s,m=compare_common_dates(old,new,official,[0,1])
    assert s['annualized_excess_delta']==pytest.approx(0.)
    assert len(s['new_excluded_dates'])==1
    assert s['old_saved_benchmark_max_difference']==.0001
    assert (m.excess_return_new==m.excess_return_old).all()


def test_comparison_rejects_internal_calendar_gap():
    new=return_fixture();old=new.drop(index=20)
    with pytest.raises(DiagnosticError,match='Internal calendar'):
        compare_common_dates(old,new,new[['trade_date','benchmark_return']],[0,1])


def test_monthly_cost_addback_is_compounded_and_partial_year_is_labelled():
    daily=return_fixture();daily['nav']=1000*(1+daily.portfolio_return).cumprod()
    daily['transaction_cost']=.1;daily['cash']=0.;daily['single_sided_turnover']=.01
    s,m,y=performance_and_costs(daily,1000.,[0,1])
    assert s['cost_drag_annualized']>0
    assert s['cost_drag_annualized']==pytest.approx(m.cost_drag.mean()*12)
    assert y.partial_year.iloc[0]
    assert y.period_excess_difference.iloc[0]==pytest.approx((1+daily.portfolio_return).prod()-(1+daily.benchmark_return).prod())


def test_attribution_closes_with_cash_and_actual_intraday_trade():
    daily,trades,market,costs=ledger_fixture()
    prior=pd.Timestamp('2020-01-01')
    # At least 3 observations for one industry + one style regression.
    extra=market.loc[market.instrument.eq('B')].assign(instrument='C',close=10.,vwap=10.)
    history=pd.DataFrame({'trade_date':prior,'instrument':['A','B','C'],'close':10.,'vwap':10.,
        'daily_amount_cny':1e6,'volatility':.02,'exchange_can_buy':True,'exchange_can_sell':True})
    market=pd.concat([history,market,extra],ignore_index=True)
    path=reconstruct_d11(daily,trades,market,1000.,costs)
    exposures=pd.DataFrame([(d,n,1.,s,100.,1/3) for d in [prior,pd.Timestamp('2020-01-02')]
        for n,s in [('A',-1.),('B',0.),('C',1.)]],columns=['trade_date','instrument','industry_one','style_x','market_cap','benchmark_weight'])
    d,m,_,_,summary=attribute_d11(daily,path,market,exposures,['industry_one'],['style_x'])
    assert d.accounting_residual.abs().max()<1e-12
    assert m[['industry','style','specific_net','benchmark_bridge','unmodelled_bridge']].sum(axis=1).iloc[0]==pytest.approx(m.excess_return.iloc[0])
    assert summary['annualized_contributions']['excess_return']==pytest.approx(m.excess_return.mean()*12)


def test_full_diagnostic_saved_run_real_solver_stress_and_hashes(tmp_path,monkeypatch):
    from test_d11_pipeline import synthetic_data
    from scripts import d11_pipeline as pipeline
    config=synthetic_data(tmp_path,monkeypatch)
    (tmp_path/'config/d11_baseline.json').write_text('{}')
    def exposures(access,cfg,w,signals):
        e=w.loc[w.trade_date.isin(signals)].rename(columns={'weight':'benchmark_weight'}).assign(industry_one=1.,market_cap=100.)
        return e,pd.DataFrame({'passed':[True]}),pd.DataFrame({'passed':[True]}),['industry_one'],[],pd.DataFrame()
    monkeypatch.setattr(pipeline,'build_oos_exposures',exposures)
    source=pipeline.run_once(tmp_path,config,{},'source_v1')
    source_bytes={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    daily=pd.read_parquet(source/'daily.parquet')
    daily.to_parquet(tmp_path/'reports/d9_v2_main_daily.parquet',index=False)
    def inputs(access,cfg,saved,output):
        w=pd.concat([pd.read_parquet(tmp_path/'data/processed/design/benchmark_weights.parquet'),
                     pd.read_parquet(tmp_path/'data/processed/oos_v2/benchmark_weights.parquet')])
        flags=w[['trade_date','instrument']].assign(is_st=False,is_listed_under_120d=False,is_below_amount_floor=False)
        market=pipeline.prepare_oos_market()[0]
        calendar=pd.DatetimeIndex(w.trade_date.unique()).sort_values()
        ex=w.rename(columns={'weight':'benchmark_weight'}).assign(industry_one=1.,market_cap=100.)
        official=daily[['trade_date','benchmark_return']]
        return market,ex,flags,calendar,official,['industry_one'],[]
    monkeypatch.setattr(diagnostic,'load_inputs',inputs)
    monkeypatch.setattr(diagnostic,'preflight',lambda *a:{'synthetic_fixture':True})
    monkeypatch.setattr(diagnostic,'verify_repository',lambda *a:{})
    destination=diagnostic.run_once(tmp_path,config,'source_v1','diagnostic_v1')
    result=diagnostic.verify_diagnostic(destination)
    assert result['verified']
    s=json.loads((destination/'summary.json').read_text())
    assert s['stress']['baseline_replay']['annualized_excess_return']==pytest.approx(s['performance']['net']['annualized_excess_return'])
    assert s['stress']['impact2x']['annualized_excess_return']!=s['stress']['baseline_replay']['annualized_excess_return']
    assert s['model_training_performed'] is False and s['new_oos_validation'] is False
    assert s['accounting']['passed']
    assert source_bytes=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError): diagnostic.run_once(tmp_path,config,'source_v1','diagnostic_v1')
    (destination/'yearly_ic.tsv').write_text('tampered')
    with pytest.raises(DiagnosticError,match='hash mismatch'): diagnostic.verify_diagnostic(destination)


def test_missing_outputs_fail_before_running(tmp_path):
    with pytest.raises(DiagnosticError,match='original computer'):
        diagnostic.check_source(tmp_path)


def test_daily_exposure_loader_matches_existing_weekly_builder(tmp_path,monkeypatch):
    """Exercise actual file/date loading and all six styles, stubbing only Qlib quotes."""
    from test_d11_pipeline import synthetic_data
    from scripts import d11_pipeline as pipeline, d9_pipeline as old_builder
    from src.data_access import ResearchDataAccess
    from src.d11_io import D11Output
    config=synthetic_data(tmp_path,monkeypatch)
    market=pipeline.prepare_oos_market()[0].copy()
    history=pd.Timestamp(config['windows']['oos_start'])-pd.Timedelta(days=config['d6']['history_buffer_calendar_days'])
    market=market.loc[market.trade_date.ge(history)].copy()
    dates=pd.DatetimeIndex(market.trade_date.unique()).sort_values()
    names=sorted(market.instrument.unique())
    t=market.trade_date.map({d:i for i,d in enumerate(dates)}).to_numpy()
    n=market.instrument.map({s:i for i,s in enumerate(names)}).to_numpy()
    market['close']=10*np.exp(t*(n+1)*.00002+.01*np.sin(t/(n+2))+.02*np.sin(t/7))
    market['vwap']=market.close
    basic=market[['trade_date','instrument']].assign(total_mv=10000+n*2000+t,turnover_rate=1+n*.03+.1*np.sin(t/5))
    for year,g in basic.groupby(basic.trade_date.dt.year):
        g.to_parquet(tmp_path/f'data/raw/tushare/daily_basic_{year}.parquet',index=False)
    official=pd.DataFrame({'trade_date':dates,'close':1000*np.exp(np.arange(len(dates))*.0001+.01*np.sin(np.arange(len(dates))/7))})
    official.to_parquet(tmp_path/'data/raw/tushare/index_daily_H00905_CSI.parquet',index=False)
    official['benchmark_return']=official.close.pct_change(fill_method=None)
    mapping=pd.DataFrame({'instrument':names,'l1_code':'one','industry_name':'one','industry_factor':'industry_one'})
    catalog=mapping[['l1_code','industry_name','industry_factor']].drop_duplicates()
    mapping.to_parquet(tmp_path/config['d6']['industry_mapping_file'],index=False)
    catalog.to_parquet(tmp_path/config['d6']['industry_catalog_file'],index=False)
    monkeypatch.setattr(diagnostic,'prepare_oos_market',lambda *a:(market,pd.DataFrame()))
    monkeypatch.setattr(old_builder,'require_industry_files',lambda *a:(mapping,catalog))
    monkeypatch.setattr(old_builder,'load_close_prices',lambda *a:market[['trade_date','instrument','close']])
    monkeypatch.setattr(old_builder,'load_benchmark_returns',lambda *a:official)
    monkeypatch.setattr(old_builder,'load_daily_basic',lambda *a:basic)
    weights=pd.read_parquet(tmp_path/'data/processed/oos_v2/benchmark_weights.parquet')
    signals=pd.DatetimeIndex(weights.loc[weights.trade_date.dt.weekday.eq(4),'trade_date'].unique())
    access=ResearchDataAccess(tmp_path)
    saved=old_builder.build_oos_exposures(access,config,weights,signals)[0]
    output=D11Output(tmp_path,'loader_v1')
    _,exposure,_,_,_,industries,styles=diagnostic.load_inputs(access,config,saved,output)
    assert len(styles)==6 and industries==['industry_one']
    assert exposure.trade_date.min()<pd.Timestamp(config['windows']['oos_start'])
    assert json.loads((output.path/'exposure_parity.json').read_text())['passed']


@pytest.mark.parametrize('name',['../outside','a/b','', 'x'*81])
def test_diagnostic_path_rejects_escape(tmp_path,name):
    with pytest.raises(DiagnosticError): diagnostic.run_path(tmp_path,name)

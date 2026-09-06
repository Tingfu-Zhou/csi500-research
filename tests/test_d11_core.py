"""Causal counterfactuals: alter future labels/quotes/policy, hold T orders fixed."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.d3_core import CostModel
from src.d4_core import build_forward_excess_labels
from src.d11_core import (CausalityError, build_inference_frame, training_at, fit_predict,
                          calibration_at, signal_eligibility, freeze_orders, execute_orders,
                          simulate_causal, optimize_at_signal)
from src.d7_core import OptimizationLimits, PortfolioOptimizationError


def test_e01_future_endpoint_loss_drops_old_inner_join_but_not_inference():
    dates = pd.bdate_range('2020-12-21', periods=12)
    weights = pd.DataFrame([(d,n,.5) for d in dates for n in ['A','B']], columns=['trade_date','instrument','weight'])
    features = weights.rename(columns={'trade_date':'sample_date'}).assign(f=1.)
    prices = weights[['trade_date','instrument']].assign(vwap=10.)
    samples = dates[[0,4,8,11]]
    labels, _ = build_forward_excess_labels(prices,weights,samples,dates,[5])
    altered = prices.copy()
    altered.loc[altered.trade_date.eq(dates[6]) & altered.instrument.eq('A'),'vwap'] = np.nan
    future_labels, _ = build_forward_excess_labels(altered,weights,samples,dates,[5])
    assert len(labels.merge(features,on=['sample_date','instrument'])) > len(future_labels.merge(features,on=['sample_date','instrument']))
    inference = build_inference_frame(weights,features,['f'],samples)
    assert len(inference) == 8  # Includes incomplete final/year-end label windows.
    assert inference.sample_date.max() == dates[-1]
    pd.testing.assert_frame_equal(inference,build_inference_frame(weights,features,['f'],samples))


def model_fixture():
    rng = np.random.default_rng(8)
    dates = pd.to_datetime(['2016-10-07','2016-10-14','2016-10-21','2016-12-30','2017-12-29'])
    weights = pd.DataFrame([(d,str(n),1/80) for d in dates for n in range(80)], columns=['trade_date','instrument','weight'])
    features = weights.rename(columns={'trade_date':'sample_date'}).assign(f=rng.normal(size=len(weights)))
    labels = features[['sample_date','instrument']].assign(excess_return=features.f*.02, label_end_timestamp=features.sample_date+pd.Timedelta(days=8))
    inference = build_inference_frame(weights,features,['f'],dates)
    return inference,labels


def test_e01_future_labels_cannot_change_fitted_prediction_or_training():
    inference, labels = model_fixture()
    altered = labels.copy()
    altered.loc[altered.label_end_timestamp > '2016-12-31','excess_return'] = 999.
    altered = altered.loc[~(altered.sample_date.dt.year.eq(2017) & altered.instrument.eq('3'))]
    train = training_at(inference,labels,'2016-12-31','2014-01-01')
    pd.testing.assert_frame_equal(train,training_at(inference,altered,'2016-12-31','2014-01-01'))
    assert train.sample_date.max() == pd.Timestamp('2016-10-21')
    pred = inference.loc[inference.sample_date.dt.year.eq(2017)].copy()
    pred.loc[pred.instrument.eq('3'),['f','feature_available']] = [np.nan,False]
    result,_ = fit_predict(train,pred,['f'])
    counter,_ = fit_predict(training_at(inference,altered,'2016-12-31','2014-01-01'),pred,['f'])
    pd.testing.assert_frame_equal(result,counter)
    assert len(result) == 80
    abstain = result.loc[result.instrument.eq('3')].iloc[0]
    assert np.isnan(abstain.prediction) and abstain.score_z == 0


def test_e01_calibration_only_uses_matured_labels():
    inference,labels = model_fixture()
    history = inference.assign(score_z=inference.f)
    expected = calibration_at(history,labels,'2016-12-31')
    future = labels.copy()
    future.loc[future.label_end_timestamp > '2016-12-31','excess_return'] = 1e6
    assert calibration_at(history,future,'2016-12-31') == expected
    assert expected['coefficient'] == pytest.approx(.02)


def simulation_fixture():
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/config.yaml').read_text())
    config['windows'].update(oos_start='2020-01-03',oos_end='2020-01-08')
    config['d7'].update(stock_active_limit=.5,industry_active_limit=1.,style_active_limit=1.,single_sided_turnover_limit=1.)
    config['d8']['main_capital_cny'] = 1000.
    calendar = pd.bdate_range('2020-01-02','2020-01-08')
    market = pd.DataFrame([(d,n) for d in calendar for n in ['A','B']],columns=['trade_date','instrument'])
    market = market.assign(close=10.,vwap=10.,daily_amount_cny=1e6,volatility=.02,exchange_can_buy=True,exchange_can_sell=True)
    flags = market[['trade_date','instrument']].assign(is_st=False,is_listed_under_120d=False,is_below_amount_floor=False)
    signals = pd.to_datetime(['2020-01-03','2020-01-07'])
    pred = market.loc[market.trade_date.isin(signals),['trade_date','instrument']].rename(columns={'trade_date':'sample_date'})
    pred['expected_5d_excess'] = [.02,0,0,.02]
    exposures = pred.rename(columns={'sample_date':'trade_date'}).assign(benchmark_weight=.5)
    return market,pred,exposures,flags,calendar,config


def test_e04_future_quotes_and_e03_future_policy_do_not_change_signal_orders():
    market,pred,exp,flags,cal,config = simulation_fixture()
    base = simulate_causal(market,pred,exp,flags,cal,config,[],[])
    future = market.copy()
    future.loc[future.trade_date > '2020-01-03',['close','vwap']] = 30.
    future.loc[future.trade_date > '2020-01-03','exchange_can_buy'] = False
    future_flags = flags.copy()
    future_flags.loc[future_flags.trade_date > '2020-01-03','is_st'] = True
    # Only one decision: later realized states may legitimately change later decisions.
    first = pred.loc[pred.sample_date.eq(pd.Timestamp('2020-01-03'))]
    other = simulate_causal(future,first,exp, future_flags,cal,config,[],[])
    pd.testing.assert_frame_equal(base.targets.loc[base.targets.signal_date.eq(pd.Timestamp('2020-01-03'))].reset_index(drop=True),other.targets)
    pd.testing.assert_frame_equal(base.orders.loc[base.orders.signal_date.eq(pd.Timestamp('2020-01-03'))].reset_index(drop=True),other.orders)
    assert other.trades.executed_units.eq(0).all()
    assert base.trades.executed_units.abs().sum() > 0


def test_e03_prior_session_policy_and_unknown_names_fail_closed():
    _,_,_,flags,cal,_ = simulation_fixture()
    flags.loc[flags.trade_date.eq(pd.Timestamp('2020-01-03')),'is_st'] = True
    result = signal_eligibility(flags,pd.Timestamp('2020-01-03'),pd.Index(['A','B','NEW']),cal)
    assert result.signal_can_buy.tolist() == [True,True,False]
    assert result.eligibility_asof.eq(pd.Timestamp('2020-01-02')).all()
    with pytest.raises(CausalityError,match='No prior session'):
        signal_eligibility(flags,cal[0],pd.Index(['A']),cal)


def order_fixture():
    names = pd.Index(['A','B'],name='instrument')
    target,marks,allowed = pd.Series([.5,.5],index=names),pd.Series(10.,index=names),pd.Series(True,index=names)
    orders = freeze_orders(target,pd.Series(dtype=float),1000.,marks,allowed,'2020-01-03','2020-01-06')
    day = pd.DataFrame({'vwap':[20.,20.],'daily_amount_cny':1e9,'volatility':.02,'exchange_can_buy':True,'exchange_can_sell':True},index=names)
    return orders,day


def test_e04_fixed_quantity_gap_up_cash_scale_and_gap_down_no_enlargement():
    orders,day = order_fixture()
    assert orders.order_units.tolist() == [50.,50.]
    units,cash,trades = execute_orders(pd.Series(dtype=float),1000.,orders,day,'2020-01-06',CostModel.zero_cost(1.))
    assert units.tolist() == pytest.approx([25.,25.])
    assert cash == pytest.approx(0.)
    day.vwap = 5.
    units,cash,_ = execute_orders(pd.Series(dtype=float),1000.,orders,day,'2020-01-06',CostModel.zero_cost(1.))
    assert units.tolist() == pytest.approx([50.,50.])
    assert cash == pytest.approx(500.)


@pytest.mark.parametrize('blocked',['exchange_can_buy','missing_vwap','participation'])
def test_fill_constraints_do_not_rewrite_orders(blocked):
    orders,day = order_fixture()
    if blocked == 'exchange_can_buy': day.exchange_can_buy = False
    elif blocked == 'missing_vwap': day.vwap = np.nan
    else: day.daily_amount_cny = 200.
    units,cash,fills = execute_orders(pd.Series(dtype=float),1000.,orders,day,'2020-01-06',CostModel.zero_cost(.05))
    assert fills.order_units.tolist() == [50.,50.]
    assert fills.executed_units.tolist() == pytest.approx([.5,.5] if blocked == 'participation' else [0.,0.])
    assert cash >= 0


def test_sell_block_carries_actual_inventory_to_next_decision():
    market,pred,exp,flags,cal,config = simulation_fixture()
    market.loc[market.trade_date >= '2020-01-06','exchange_can_buy'] = False
    result = simulate_causal(market,pred,exp,flags,cal,config,[],[])
    second = result.audits.iloc[1]
    assert second.actual_holdings == 0 and second.pretrade_cash_weight == 1.
    assert not second.initial_construction_exempt
    assert result.targets.loc[result.targets.signal_date.eq(pd.Timestamp('2020-01-07'))].pretrade_weight.eq(0).all()


def test_sell_proceeds_costs_and_full_exit_inventory():
    orders,day = order_fixture()
    orders.order_units = [-50.,50.]
    config = simulation_fixture()[-1]
    units,cash,trades = execute_orders(pd.Series({'A':50.}),0.,orders,day,'2020-01-06',CostModel.from_config(config))
    assert 'A' not in units.index and units['B'] < 50 and cash >= 0
    assert trades.loc[trades.instrument.eq('A'),'stamp_duty_cost'].iloc[0] > 0
    assert trades.loc[trades.instrument.eq('B'),'stamp_duty_cost'].iloc[0] == 0
    assert 1000 - (units*20).sum() - cash == pytest.approx(trades.total_cost.sum(),abs=1e-8)


def test_infeasible_policy_stops_without_fallback():
    idx = pd.Index(['A','B'])
    with pytest.raises(PortfolioOptimizationError,match='no fallback'):
        optimize_at_signal(pd.Series(0.,index=idx),pd.Series(.5,index=idx),pd.Series(dtype=float),
                           pd.DataFrame(index=idx),pd.Series(False,index=idx),[],[],
                           OptimizationLimits(.1,1.,1.,.15),CostModel.zero_cost(1.),'2020-01-06',initial=True)


def test_unpriceable_signal_target_is_rejected():
    with pytest.raises(CausalityError,match='no known mark'):
        freeze_orders(pd.Series({'A':1.}),pd.Series(dtype=float),100.,pd.Series(dtype=float),
                      pd.Series({'A':True}),'2020-01-03','2020-01-06')


def test_e04_reproduces_legacy_execution_day_drift_dependency():
    from scripts.d7_pipeline import build_execution_price_matrix, holding_returns_between
    from src.d7_core import drift_pretrade_weights
    dates = pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'])
    prices = pd.DataFrame([(d,n,10.,10.) for d in dates for n in ['A','B']],
                          columns=['trade_date','instrument','vwap','close'])
    prior = pd.Series({'A':.5,'B':.5})
    def drift(p):
        returns = holding_returns_between(build_execution_price_matrix(p),dates[0],dates[-1],prior.index)
        return drift_pretrade_weights(prior,returns)
    assert drift(prices)['A'] == pytest.approx(.5)
    prices.loc[prices.trade_date.eq(dates[-1]) & prices.instrument.eq('A'),'vwap'] = 20.
    assert drift(prices)['A'] == pytest.approx(2/3)


def test_e03_reproduces_legacy_mixed_policy_while_exchange_is_open():
    from scripts.d3_pipeline import prepare_markets
    market,_,_,flags,_,_ = simulation_fixture()
    flags = flags.assign(can_buy=True,can_sell=True)
    flags.loc[flags.trade_date.eq(pd.Timestamp('2020-01-06')),'can_buy'] = False
    strategy,replication = prepare_markets(market,flags)
    assert strategy.loc[strategy.trade_date.eq(pd.Timestamp('2020-01-06'))].can_buy.eq(False).all()
    assert replication.loc[replication.trade_date.eq(pd.Timestamp('2020-01-06'))].can_buy.eq(True).all()

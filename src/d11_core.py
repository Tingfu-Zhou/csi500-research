"""D11 causal repair path. Frozen D1-D10 modules remain historical evidence.

Inference is label-independent. Targets and orders are fixed at T end-of-day.
The next session may change fills, never the already submitted order quantity.
Units below are Qlib adjusted units, not exchange round lots.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.d3_core import CostModel, trade_cost
from src.d4_core import cross_sectional_zscore
from src.d5_core import rank_gaussianize, fit_return_calibration
from src.d7_core import OptimizationLimits, PortfolioOptimizationError


class CausalityError(ValueError):
    pass


def unique(frame, keys, name):
    if frame.duplicated(keys).any():
        raise CausalityError(f"{name}: duplicate keys {keys}")


def build_inference_frame(weights, features, feature_names, sample_dates):
    """All T members, LEFT joined only to T features; never accepts labels."""
    keys = ['sample_date', 'instrument']
    dates = pd.DatetimeIndex(sample_dates).normalize()
    base = weights.loc[weights.trade_date.isin(dates),
                       ['trade_date', 'instrument', 'weight']].rename(
                           columns={'trade_date': 'sample_date', 'weight': 'benchmark_weight'})
    unique(base, keys, 'inference universe')
    unique(features, keys, 'features')
    result = base.merge(features[keys + list(feature_names)], on=keys, how='left', validate='one_to_one')
    result[feature_names] = result[feature_names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    result['feature_available'] = result[feature_names].notna().any(axis=1)
    result['feature_asof'] = result.sample_date
    return result.sort_values(keys).reset_index(drop=True)


def training_at(inference, labels, cutoff, start):
    """Only matured labels may enter fitting. Missing labels cannot filter inference."""
    cutoff, start = pd.Timestamp(cutoff), pd.Timestamp(start)
    required = ['sample_date', 'instrument', 'label_end_timestamp', 'excess_return']
    unique(labels, ['sample_date', 'instrument'], 'labels')
    past = labels.loc[labels.label_end_timestamp.le(cutoff)
                      & labels.sample_date.between(start, cutoff), required]
    result = inference.loc[inference.sample_date.between(start, cutoff)].merge(
        past, on=['sample_date', 'instrument'], how='inner', validate='one_to_one')
    result = result.loc[result.feature_available & np.isfinite(result.excess_return)].copy()
    if result.empty:
        raise CausalityError('No matured training data')
    if (result.label_end_timestamp > cutoff).any():
        raise CausalityError('Training labels crossed cutoff')
    return result


def fit_predict(train, inference, feature_names, *, seed=42, model_path=None):
    import lightgbm as lgb
    target = cross_sectional_zscore(train, 'excess_return')
    usable = target.notna()
    if not usable.any():
        raise CausalityError('No nonconstant training cross sections')
    model = lgb.LGBMRegressor(random_state=int(seed))
    model.fit(train.loc[usable, feature_names], target.loc[usable])
    result = inference[['sample_date', 'instrument', 'feature_available', 'feature_asof']].copy()
    result['prediction'] = np.nan
    good = inference.feature_available
    if good.any():
        result.loc[good, 'prediction'] = model.predict(inference.loc[good, feature_names])
        if not np.isfinite(result.loc[good, 'prediction']).all():
            raise CausalityError('Nonfinite predictions')
    # All-missing features and constant cross sections abstain, using only T data.
    result['score_z'] = rank_gaussianize(result).fillna(0.0)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))
    return result, model.get_params()


def calibration_at(history, labels, cutoff):
    cutoff = pd.Timestamp(cutoff)
    unique(labels, ['sample_date', 'instrument'], 'calibration labels')
    mature = labels.loc[labels.label_end_timestamp.le(cutoff),
                        ['sample_date', 'instrument', 'label_end_timestamp', 'excess_return']]
    past = history.loc[history.sample_date.le(cutoff) & history.feature_available].merge(
        mature, on=['sample_date', 'instrument'], how='inner', validate='one_to_one')
    result = fit_return_calibration(past)
    result['last_label_end'] = str(past.label_end_timestamp.max().date())
    assert past.label_end_timestamp.max() <= cutoff
    return result


def signal_eligibility(flags, signal_date, instruments, calendar):
    """Conservative one-session lag for ST/age/liquidity policy, missing=False.

    Exchange suspension/limit flags are deliberately excluded here. Those are
    realized fill constraints. The source-date column makes timing auditable.
    """
    prior = pd.DatetimeIndex(calendar)[pd.DatetimeIndex(calendar) < pd.Timestamp(signal_date)]
    if len(prior) == 0:
        raise CausalityError('No prior session for eligibility')
    asof = prior.max()
    cols = ['is_st', 'is_listed_under_120d', 'is_below_amount_floor']
    unique(flags, ['trade_date', 'instrument'], 'eligibility flags')
    frame = flags.loc[flags.trade_date.eq(asof)].set_index('instrument').reindex(instruments)
    known = frame[cols].notna().all(axis=1)
    allowed = known & frame[cols].eq(False).all(axis=1)
    return pd.DataFrame({'signal_can_buy': allowed, 'eligibility_known': known,
                         'eligibility_asof': asof}, index=instruments)


def optimize_at_signal(mu, benchmark, pretrade, exposures, buy_allowed,
                       industries, styles, limits, costs, execution_date, *, initial=False):
    """Optimize using actual T holdings + cash. Costs are side-specific.

    Initial construction from cash is explicitly exempt from the turnover cap,
    with all opening costs charged. Later cash is not silently renormalized.
    """
    import cvxpy as cp
    limits.validate()
    idx = benchmark.index
    if idx.has_duplicates or pretrade.index.has_duplicates:
        raise CausalityError('Duplicate portfolio instruments')
    wb, w0 = benchmark.astype(float), pretrade.astype(float)
    factors = exposures.reindex(idx)[list(industries) + list(styles)].to_numpy(float)
    if (not np.isfinite(wb).all() or not np.isfinite(w0).all()
            or not np.isfinite(mu.reindex(idx)).all() or not np.isfinite(factors).all()
            or (wb < 0).any() or (w0 < 0).any()
            or not np.isclose(wb.sum(), 1, atol=1e-8, rtol=0)
            or w0.sum() > 1 + 1e-8):
        raise CausalityError('Invalid optimization inputs')
    if initial and w0.sum() > 1e-10:
        raise CausalityError('Initial exemption requires cash-only portfolio')
    old = w0.reindex(idx, fill_value=0.0).to_numpy()
    forced = float(w0.loc[~w0.index.isin(idx)].sum())
    w = cp.Variable(len(idx))
    active = w - wb.to_numpy()
    total_l1 = cp.norm1(w - old) + forced
    constraints = [cp.sum(w) == 1, w >= 0,
                   active <= limits.stock_active, active >= -limits.stock_active]
    for cols, bound in [(industries, limits.industry_active), (styles, limits.style_active)]:
        if cols:
            f = exposures.reindex(idx)[list(cols)].to_numpy(float).T @ active
            constraints += [f <= bound, f >= -bound]
    allowed = buy_allowed.reindex(idx).eq(True).to_numpy()
    if (~allowed).any():
        constraints.append(w[~allowed] <= old[~allowed])
    if not initial:
        constraints.append(0.5 * total_l1 - forced <= limits.single_sided_turnover)
    buys, sells = cp.sum(cp.pos(w - old)), cp.sum(cp.pos(old - w)) + forced
    fee = (costs.commission_rate + costs.transfer_fee_rate) * (buys + sells)
    fee += costs.stamp_rate(execution_date) * sells
    problem = cp.Problem(cp.Maximize(mu.reindex(idx).to_numpy() @ active - fee), constraints)
    problem.solve(solver='CLARABEL')
    if problem.status != 'optimal' or w.value is None:
        raise PortfolioOptimizationError(f'D11 strict solver status={problem.status}; no fallback')
    target = pd.Series(np.maximum(w.value, 0.0), index=idx)
    # Clip numerical buy dust on explicitly forbidden names, never relax policy.
    target.loc[~allowed] = np.minimum(target.loc[~allowed], old[~allowed])
    a = target - wb
    turnover = 0.5 * (float((target - old).abs().sum()) + forced)
    audit = {'solver_status': problem.status, 'fallback_used': False,
             'initial_construction_exempt': bool(initial), 'pretrade_cash_weight': float(1-w0.sum()),
             'maximum_stock_active': float(a.abs().max()),
             'maximum_industry_active': float((exposures.reindex(idx)[list(industries)].T @ a).abs().max()) if industries else 0.,
             'maximum_style_active': float((exposures.reindex(idx)[list(styles)].T @ a).abs().max()) if styles else 0.,
             'target_total_turnover': turnover, 'mandatory_turnover': forced,
             'target_active_turnover': max(turnover-forced, 0.),
             'target_weight_sum_error': abs(float(target.sum())-1),
             'prohibited_buy_weight': float((target.loc[~allowed] - old[~allowed]).clip(lower=0).sum())}
    ok = [audit['target_weight_sum_error'] <= limits.tolerance,
          audit['maximum_stock_active'] <= limits.stock_active + limits.tolerance,
          audit['maximum_industry_active'] <= limits.industry_active + limits.tolerance,
          audit['maximum_style_active'] <= limits.style_active + limits.tolerance,
          audit['prohibited_buy_weight'] <= limits.tolerance,
          initial or audit['target_active_turnover'] <= limits.single_sided_turnover + limits.tolerance]
    if not all(ok):
        raise PortfolioOptimizationError(f'D11 target audit failed: {audit}')
    audit['passed'] = True
    return target, audit


def freeze_orders(target, units, cash, marks, buy_allowed, signal_date, execution_date):
    """Fix quantities from signal-time NAV and known marks; no execution quote."""
    idx = units.index.union(target.index)
    held = units.reindex(idx, fill_value=0.).astype(float)
    price = marks.reindex(idx).astype(float)
    if not np.isfinite(cash) or cash < 0 or (held < 0).any():
        raise CausalityError('Invalid signal portfolio')
    needed = held.gt(0) | target.reindex(idx, fill_value=0).gt(1e-12)
    if (~(price.gt(0) & np.isfinite(price)) & needed).any():
        raise CausalityError('Signal target/holding has no known mark')
    nav = float((held * price.fillna(0)).sum() + cash)
    if not np.isfinite(nav) or nav <= 0:
        raise CausalityError('Invalid signal NAV')
    tw = target.reindex(idx, fill_value=0)
    desired_units = tw * nav / price.where(price.gt(0))
    quantity = desired_units.fillna(0) - held
    prohibited = ~buy_allowed.reindex(idx).eq(True)
    if ((quantity > 1e-8) & prohibited).any():
        raise CausalityError('Target attempts policy-prohibited buy')
    quantity.loc[prohibited] = quantity.loc[prohibited].clip(upper=0)
    out = pd.DataFrame({'instrument': idx, 'signal_date': pd.Timestamp(signal_date),
                        'execution_date': pd.Timestamp(execution_date), 'signal_price': price.to_numpy(),
                        'signal_nav': nav, 'signal_units': held.to_numpy(),
                        'target_weight': tw.to_numpy(), 'order_units': quantity.to_numpy(),
                        'signal_can_buy': buy_allowed.reindex(idx).eq(True).to_numpy()})
    return out.loc[(out.order_units.abs() * out.signal_price.fillna(0)).gt(nav*1e-12)].reset_index(drop=True)


def execute_orders(units, cash, orders, day, date, costs):
    """Fill fixed quantities; realized exchange/volume affect fills only."""
    if orders.empty:
        return units.copy(), float(cash), pd.DataFrame()
    unique(orders, ['instrument'], 'orders')
    if not orders.execution_date.eq(pd.Timestamp(date)).all() or not orders.signal_date.lt(pd.Timestamp(date)).all():
        raise CausalityError('Invalid T+1 order schedule')
    idx = pd.Index(orders.instrument)
    q = day.reindex(idx)
    price = pd.to_numeric(q.vwap, errors='coerce')
    amount = pd.to_numeric(q.daily_amount_cny, errors='coerce')
    vol = pd.to_numeric(q.volatility, errors='coerce')
    wanted = orders.set_index('instrument').order_units
    valid = price.gt(0) & np.isfinite(price) & amount.gt(0) & np.isfinite(amount) & vol.ge(0) & np.isfinite(vol)
    capacity = (amount * costs.participation_limit / price).where(valid, 0.)
    filled = pd.Series(0., index=idx)
    held = units.reindex(units.index.union(idx), fill_value=0.).copy().astype(float)
    out = orders.set_index('instrument').copy()
    out['exchange_can_buy'] = q.exchange_can_buy.eq(True)
    out['exchange_can_sell'] = q.exchange_can_sell.eq(True)
    out['execution_price'] = price
    for name in idx[wanted < 0]:
        if valid.loc[name] and q.loc[name, 'exchange_can_sell'] == True:
            filled.loc[name] = -min(-wanted.loc[name], float(held.loc[name]), capacity.loc[name])
    def fees(name, quantity):
        return trade_cost(float(quantity * price.loc[name]) if quantity else 0.,
                          vol.loc[name], amount.loc[name], date, costs)
    sell_cost = sum(fees(name, filled.loc[name])['total_cost'] for name in idx[filled < 0])
    cash = float(cash - (filled * price.fillna(0)).sum() - sell_cost)
    if cash < -1e-8:
        raise CausalityError('Sell costs exhausted available cash')
    candidates = pd.Series(0., index=idx)
    for name in idx[wanted > 0]:
        if not bool(out.loc[name, 'signal_can_buy']):
            raise CausalityError('Submitted buy without signal eligibility')
        if valid.loc[name] and q.loc[name, 'exchange_can_buy'] == True:
            candidates.loc[name] = min(wanted.loc[name], capacity.loc[name])
    buying = idx[candidates > 0]
    buy_notional = (candidates.loc[buying] * price.loc[buying]).to_numpy()
    buy_impact = (costs.impact_coefficient * vol.loc[buying].to_numpy()
                  * np.sqrt(buy_notional / amount.loc[buying].to_numpy()))
    def need(scale):
        return float((buy_notional * scale * (1 + costs.commission_rate
                     + costs.transfer_fee_rate + buy_impact * np.sqrt(scale))).sum())
    scale = 1.
    if need(1.) > max(cash, 0.):
        low, high = 0., 1.
        for _ in range(55):
            mid = (low+high)/2
            if need(mid) <= max(cash, 0.): low = mid
            else: high = mid
        scale = low
    filled.loc[buying] = candidates.loc[buying] * scale
    cash -= need(scale)
    if cash < -1e-7:
        raise CausalityError('Negative cash after fills')
    cash = max(cash, 0.)
    for n in idx:
        old = held.loc[n]
        held.loc[n] = old + filled.loc[n]
        if filled.loc[n] < 0 and abs(held.loc[n]) <= max(abs(old)*1e-12, 1e-12):
            held.loc[n] = 0.
    if (held < 0).any(): raise CausalityError('Negative inventory')
    held = held.loc[held > 0]
    out['executed_units'] = filled
    out['executed_trade_amount'] = filled * price.fillna(0)
    out['unfilled_units'] = wanted-filled
    out['cash_scale'] = scale
    out['participation_limited'] = wanted.abs().gt(capacity) & valid
    cost_rows = pd.DataFrame([fees(n, filled.loc[n]) for n in idx], index=idx)
    for col in cost_rows: out[col] = cost_rows[col]
    return held, cash, out.reset_index()


@dataclass
class Simulation:
    nav: pd.DataFrame
    targets: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    audits: pd.DataFrame


def simulate_causal(market, predictions, exposures, flags, calendar, config,
                    industries, styles, *, progress=None):
    """One daily loop: prior orders fill -> close mark -> optimize -> freeze orders."""
    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    unique(market, ['trade_date', 'instrument'], 'market')
    unique(predictions, ['sample_date', 'instrument'], 'predictions')
    unique(exposures, ['trade_date', 'instrument'], 'exposures')
    d7 = config['d7']
    limits = OptimizationLimits(d7['stock_active_limit'], d7['industry_active_limit'],
                               d7['style_active_limit'], d7['single_sided_turnover_limit'], d7['constraint_tolerance'])
    costs = CostModel.from_config(config)
    cash = float(config['d8']['main_capital_cny'])
    units, marks = pd.Series(dtype=float), pd.Series(dtype=float)
    pending = pd.DataFrame()
    targets, orders, trades, audits, navs = [], [], [], [], []
    price_days = {pd.Timestamp(k):v.set_index('instrument') for k,v in market.groupby('trade_date')}
    pred_days = {pd.Timestamp(k):v.set_index('instrument') for k,v in predictions.groupby('sample_date')}
    exposure_days = {pd.Timestamp(k):v.set_index('instrument').sort_index() for k,v in exposures.groupby('trade_date')}
    # Previous calendar session is needed even before the simulated OOS start.
    start, end = pd.Timestamp(config['windows']['oos_start']), pd.Timestamp(config['windows']['oos_end'])
    dates = calendar[(calendar >= start) & (calendar <= end)]
    for day_number, date in enumerate(dates):
        day = price_days.get(date)
        if day is None: raise CausalityError(f'No market session {date}')
        fee, turnover, signal_executed = 0., 0., None
        if not pending.empty:
            if not pending.execution_date.eq(date).all(): raise CausalityError('Missed pending execution')
            units, cash, fills = execute_orders(units,cash,pending,day,date,costs)
            trades.append(fills)
            fee = float(fills.total_cost.sum())
            turnover = .5*float(fills.executed_trade_amount.abs().sum())/float(pending.signal_nav.iloc[0])
            signal_executed = pending.signal_date.iloc[0]
            pending = pd.DataFrame()
        close = pd.to_numeric(day.close,errors='coerce').where(lambda x:(x>0)&np.isfinite(x)).dropna()
        marks = marks.reindex(marks.index.union(close.index))
        marks.loc[close.index] = close
        if marks.reindex(units.index).isna().any(): raise CausalityError('Unpriceable actual holding')
        value = float((units * marks.reindex(units.index)).sum())
        nav = cash+value
        if not np.isfinite(nav) or nav <= 0: raise CausalityError('Invalid NAV')
        navs.append({'trade_date':date,'nav':nav,'cash':cash,'gross_exposure':value/nav,
                     'transaction_cost':fee,'single_sided_turnover':turnover,'signal_date':signal_executed})
        if date not in pred_days or day_number+1 == len(dates): continue
        next_date = dates[day_number+1]
        if date not in exposure_days: raise CausalityError(f'Missing signal exposures {date}')
        cross = exposure_days[date]
        prediction = pred_days[date]
        if set(prediction.index) != set(cross.index): raise CausalityError('Inference/exposure universe mismatch')
        wb = cross.benchmark_weight.astype(float)
        wb = wb/wb.sum()
        w0 = (units*marks.reindex(units.index))/nav
        mu = prediction[config['d5']['expected_return_column']].reindex(cross.index)
        policy = signal_eligibility(flags,date,cross.index,calendar)
        known_price = marks.reindex(cross.index).gt(0)
        allowed = policy.signal_can_buy & known_price
        target, audit = optimize_at_signal(mu,wb,w0,cross,allowed,industries,styles,
                                           limits,costs,next_date,initial=len(audits)==0)
        pending = freeze_orders(target,units,cash,marks,allowed,date,next_date)
        t = pd.DataFrame({'instrument':cross.index,'signal_date':date,'execution_date':next_date,
                          'target_weight':target.to_numpy(),'benchmark_weight':wb.to_numpy(),
                          'pretrade_weight':w0.reindex(cross.index,fill_value=0).to_numpy(),
                          'expected_excess':mu.to_numpy(),'signal_can_buy':allowed.to_numpy(),
                          'eligibility_asof':policy.eligibility_asof.to_numpy(),
                          'decision_asof':date})
        targets.append(t); orders.append(pending.copy())
        audit.update(signal_date=date,execution_date=next_date,decision_asof=date,
                     eligibility_asof=policy.eligibility_asof.iloc[0],signal_nav=nav,
                     actual_holdings=len(units),prohibited_or_unknown_buy_names=int((~allowed).sum()))
        audits.append(audit)
        if progress is not None: progress(t,pending,audit)
    combine = lambda xs: pd.concat(xs,ignore_index=True) if xs else pd.DataFrame()
    return Simulation(pd.DataFrame(navs),combine(targets),combine(orders),combine(trades),pd.DataFrame(audits))

"""Post-OOS causal repair experiment; never invokes D9/D10 main or unseals data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series
from src.d4_core import build_forward_excess_labels, select_weekly_sample_dates
from src.d9_core import summarize_monthly_performance
from src.d11_core import (CausalityError, build_inference_frame, training_at, fit_predict,
                          calibration_at, simulate_causal, unique)
from src.d11_io import (D11Output, preflight, snapshot_inputs, assert_snapshot_unchanged, sha256)


def load_open_calendar(access, config, *, end_date):
    frame = access.read_parquet(Path(config['data']['tushare_raw_dir'])/'trade_cal.parquet',
                                date_col='cal_date', end_date=end_date)
    return pd.DatetimeIndex(parse_date_series(frame.loc[frame.is_open.astype(int).eq(1),'cal_date'])).normalize().sort_values().unique()


def load_weekly_alpha158(*args):
    from scripts.d9_pipeline import load_weekly_alpha158 as load
    return load(*args)


def load_vwap_by_year(*args):
    from scripts.d9_pipeline import load_vwap_by_year as load
    return load(*args)


def build_oos_exposures(*args):
    from scripts.d9_pipeline import build_oos_exposures as load
    return load(*args)


def prepare_oos_market(*args):
    from scripts.d9_pipeline import prepare_oos_market as load
    return load(*args)


def read_frame(access, path, start, end):
    frame = access.read_parquet(path, date_col='trade_date', start_date=start, end_date=end)
    frame['trade_date'] = parse_date_series(frame.trade_date)
    if frame.empty:
        raise CausalityError(f'Empty input: {path}')
    return frame


def walk_forward(inference, labels, names, config, output):
    """Rebuild 2017+ calibration history, including formerly dropped year-end rows."""
    history, audits = [], []
    start = pd.Timestamp(config['windows']['design_start'])
    oos_end = pd.Timestamp(config['windows']['oos_end'])
    first_year = pd.Timestamp(config['d4']['validation_start']).year
    for year in range(first_year, oos_end.year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=1)
        test = inference.loc[inference.sample_date.dt.year.eq(year)].copy()
        if test.empty:
            raise CausalityError(f'No inference rows in {year}')
        train = training_at(inference, labels, cutoff, start)
        pred, params = fit_predict(train, test, names, seed=config['project']['random_seed'],
                                   model_path=output.target(f'lightgbm_{year}.txt'))
        # First history year is score-only; not traded or assigned a fitted return.
        calibration = calibration_at(pd.concat(history, ignore_index=True), labels, cutoff) if history else None
        coefficient = calibration['coefficient'] if calibration is not None else 0.0
        pred[config['d5']['expected_return_column']] = pred.score_z * coefficient
        pred['model_cutoff'] = cutoff
        pred['calibration_cutoff'] = cutoff if calibration is not None else pd.NaT
        pred['benchmark_weight'] = test.benchmark_weight.to_numpy()
        # Evaluation is separate and LEFT joined, after scores/returns are fixed.
        coverage = pred[['sample_date', 'instrument']].merge(
            labels[['sample_date', 'instrument', 'label_end_timestamp']],
            how='left', on=['sample_date', 'instrument'], validate='one_to_one')
        audits.append({'prediction_year': year, 'model_cutoff': cutoff,
                       'last_training_label_end': train.label_end_timestamp.max(),
                       'training_rows': len(train), 'inference_rows': len(pred),
                       'inference_dates': pred.sample_date.nunique(),
                       'feature_abstentions': int((~pred.feature_available).sum()),
                       'rows_without_evaluation_label': int(coverage.label_end_timestamp.isna().sum()),
                       'last_inference_date': pred.sample_date.max(),
                       'calibration': calibration, 'model_parameters': params})
        history.append(pred)
        output.parquet(f'predictions_{year}.parquet', pred)
        output.json(f'fold_{year}.json', audits[-1])
        print(f'D11 model {year}: train={len(train)}, inference={len(pred)}', flush=True)
    return pd.concat(history, ignore_index=True), audits


def run_research(access, config, output):
    start, end = pd.Timestamp(config['windows']['design_start']), pd.Timestamp(config['windows']['oos_end'])
    oos_start = pd.Timestamp(config['windows']['oos_start'])
    weights, flags = [], []
    for directory, lo, hi in [(config['data']['processed_design_dir'], start, config['windows']['design_end']),
                               (config['d9']['v2']['processed_oos_dir'], oos_start, end)]:
        weights.append(read_frame(access, Path(directory)/'benchmark_weights.parquet', lo, hi))
        flags.append(read_frame(access, Path(directory)/'tradability_flags.parquet', lo, hi))
    weights, flags = pd.concat(weights, ignore_index=True), pd.concat(flags, ignore_index=True)
    unique(weights, ['trade_date', 'instrument'], 'weights')
    calendar = load_open_calendar(access, config, end_date=end)
    expected = calendar[(calendar >= start) & (calendar <= end)]
    if len(expected.difference(pd.DatetimeIndex(weights.trade_date.unique()))):
        raise CausalityError('Benchmark weights missing trading sessions')
    dates = select_weekly_sample_dates(weights)
    instruments = sorted(weights.instrument.unique())
    features, spec = load_weekly_alpha158(access, instruments, dates, config)
    names = spec.feature.tolist()
    inference = build_inference_frame(weights, features, names, dates)
    output.parquet('inference_universe.parquet', inference.drop(columns=names))
    output.parquet('feature_spec.parquet', spec)
    del features
    vwap = load_vwap_by_year(access, instruments, start, end)
    labels, coverage = build_forward_excess_labels(vwap, weights, dates, calendar,
                                                   [int(config['label']['horizon_trade_days'])])
    del vwap
    output.parquet('labels_for_training_and_evaluation.parquet', labels)
    output.parquet('label_coverage.parquet', coverage)
    predictions, folds = walk_forward(inference, labels, names, config, output)
    del inference
    pred = predictions.loc[predictions.sample_date.between(oos_start, end)].copy()
    output.parquet('predictions.parquet', pred)
    # The final session has no in-window T+1. Keep its prediction; do not trade it.
    final_session = expected.max()
    signals = pd.DatetimeIndex(pred.sample_date.unique()).sort_values()
    executed_signals = signals[signals < final_session]
    exposures, style_audit, matrix_audit, industries, styles, _ = build_oos_exposures(
        access, config, weights, executed_signals)
    output.parquet('factor_exposures.parquet', exposures)
    output.parquet('style_audit.parquet', style_audit)
    output.parquet('matrix_audit.parquet', matrix_audit)
    # The raw enriched market contains exchange flags, but no strategy-policy gating.
    market, _ = prepare_oos_market(access, config, weights, flags)
    constrained_styles = ['style_' + name for name in config['d7']['constrained_style_factors']]
    if not set(constrained_styles).issubset(styles):
        raise CausalityError('Missing constrained styles')
    def checkpoint(target, orders, audit):
        date = pd.Timestamp(audit['signal_date']).strftime('%Y%m%d')
        output.json(f'decision_{date}.json', audit)
        # Retain the exact last decision even if a later date is infeasible.
        output.parquet(f'orders_{date}.parquet', orders)
        if pd.Timestamp(audit['signal_date']).month == 1:
            print(f'D11 decision {date}: actual holdings={audit["actual_holdings"]}', flush=True)
    result = simulate_causal(market, pred, exposures, flags, calendar, config,
                             industries, constrained_styles, progress=checkpoint)
    for name in ['targets', 'orders', 'trades', 'audits']:
        output.parquet(f'{name}.parquet', getattr(result, name))
    official = read_frame(access, Path(config['data']['tushare_raw_dir'])/'index_daily_H00905_CSI.parquet', start, end)
    unique(official, ['trade_date'], 'official benchmark')
    official = official.sort_values('trade_date')
    official['benchmark_return'] = official.close.pct_change(fill_method=None)
    daily = result.nav.merge(official[['trade_date', 'benchmark_return']], on='trade_date', how='left', validate='one_to_one')
    daily['portfolio_return'] = daily.nav / daily.nav.shift(1, fill_value=float(config['d8']['main_capital_cny'])) - 1
    if not np.isfinite(daily[['portfolio_return', 'benchmark_return']]).all().all():
        raise CausalityError('Missing or nonfinite daily returns, including first benchmark day')
    output.parquet('daily.parquet', daily)
    performance, monthly = summarize_monthly_performance(daily, newey_west_lags=config['d9']['newey_west_lags'])
    output.parquet('monthly.parquet', monthly)
    return {'classification': 'post_oos_causal_repair_diagnostic',
            'historical_decision': 'No-Go', 'new_oos_validation': False,
            'performance': performance, 'folds': len(folds),
            'prediction_dates': len(signals), 'decision_dates': len(result.audits),
            'prediction_only_dates_without_in_window_t1': [str(x.date()) for x in signals.difference(executed_signals)],
            'scope': 'main capital, final constrained portfolio; no new attribution, impact2x or Go/No-Go evaluation',
            'opening_policy': 'cash; initial construction exempt from turnover, all costs charged',
            'order_policy': 'fixed adjusted units at T end-of-day; one-session validity; sell then proportional cash scaling',
            'decision_time': 'T end-of-day, after daily inputs; eligibility source T-1',
            'limitations': ['static industry mapping', 'historical data vintages not proven point-in-time',
                            'adjusted fractional units, no round lots or dividends cash ledger',
                            'realized full-day VWAP/volume is an execution approximation',
                            'unfilled orders expire; next weekly decision uses actual holdings and cash']}


def verify_run(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['status'] != 'completed' or not manifest['inputs_unchanged']:
        raise CausalityError('Run did not complete with unchanged inputs')
    for name, digest in manifest['output_sha256'].items():
        if Path(name).name != name or sha256(path/name) != digest:
            raise CausalityError(f'Output hash mismatch: {name}')
    orders, trades = pd.read_parquet(path/'orders.parquet'), pd.read_parquet(path/'trades.parquet')
    targets, audits = pd.read_parquet(path/'targets.parquet'), pd.read_parquet(path/'audits.parquet')
    predictions = pd.read_parquet(path/'predictions.parquet')
    inference = pd.read_parquet(path/'inference_universe.parquet')
    dates = pd.DatetimeIndex(predictions.sample_date.unique())
    expected = inference.loc[inference.sample_date.isin(dates), ['sample_date', 'instrument']]
    keys = ['sample_date', 'instrument']
    pd.testing.assert_frame_equal(expected[keys].sort_values(keys).reset_index(drop=True),
                                  predictions[keys].sort_values(keys).reset_index(drop=True))
    if not (predictions.model_cutoff < predictions.sample_date).all():
        raise CausalityError('Model cutoff is not prior to inference')
    if not (orders.signal_date < orders.execution_date).all():
        raise CausalityError('Order timing failed')
    if not (targets.eligibility_asof < targets.signal_date).all():
        raise CausalityError('Eligibility timing failed')
    if (orders.loc[orders.order_units > 0, 'signal_can_buy'] != True).any():
        raise CausalityError('Policy-prohibited order')
    if (trades.executed_units.abs() > trades.order_units.abs() + 1e-8).any():
        raise CausalityError('Fill enlarged fixed order')
    if ((trades.executed_units > 0) & ~trades.exchange_can_buy).any() or ((trades.executed_units < 0) & ~trades.exchange_can_sell).any():
        raise CausalityError('Exchange-prohibited fill')
    if not audits.passed.all() or audits.fallback_used.any():
        raise CausalityError('Target constraints failed')
    daily = pd.read_parquet(path/'daily.parquet')
    if (daily.cash < 0).any() or not np.isfinite(daily.nav).all() or (daily.nav <= 0).any():
        raise CausalityError('Invalid cash/NAV')
    return {'verified': True, 'orders': len(orders), 'decisions': len(audits), 'historical_decision': 'No-Go'}


def run_once(root, config, baseline, run_id):
    checked = preflight(root, config, baseline)
    before = snapshot_inputs(root, config)
    output = D11Output(root, run_id)
    output.json('inputs_before.json', before)
    manifest = {'classification': 'post_oos_causal_repair_diagnostic', 'preflight': checked,
                'run_id': run_id, 'status': 'running', 'inputs_unchanged': False}
    output.json('started.json', manifest)
    try:
        summary = run_research(ResearchDataAccess(root), config, output)
        output.json('summary.json', summary)
        after = snapshot_inputs(root, config)
        output.json('inputs_after.json', after)
        assert_snapshot_unchanged(before, after)
        preflight(root, config, baseline)
        manifest.update(status='completed', inputs_unchanged=True)
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        output.text('failure.txt', traceback.format_exc())
        output.json('manifest.json', manifest)
        raise
    manifest['output_sha256'] = {p.name: sha256(p) for p in sorted(output.path.iterdir()) if p.is_file()}
    output.json('manifest.json', manifest)
    return output.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--run-id')
    action.add_argument('--verify-run', metavar='RUN_ID')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/'config/config.yaml').read_text())
    baseline = json.loads((ROOT/'config/d11_baseline.json').read_text())
    if args.preflight:
        print(json.dumps(preflight(ROOT, config, baseline), indent=2, ensure_ascii=False))
    elif args.verify_run:
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.verify_run):
            parser.error('Invalid run id')
        print(json.dumps(verify_run(ROOT/'reports/d11'/args.verify_run), indent=2))
    else:
        path = run_once(ROOT, config, baseline, args.run_id)
        print(json.dumps(verify_run(path), indent=2))


if __name__ == '__main__':
    main()

"""Review committed D11 JSON evidence. Does not train, simulate, or claim NAV replication."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/d11/d11_v1'
SOURCE_COMMIT = '51c280fa75b46bd718244b4b582f48ba0c1201d8'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def review():
    summary, manifest = read(RUN/'summary.json'), read(RUN/'manifest.json')
    old = read(ROOT/'reports/d9_v2_summary.json')['performance']
    p = summary['performance']
    if manifest['status'] != 'completed' or not manifest['inputs_unchanged']:
        raise ValueError('Run did not complete with unchanged inputs')
    files = sorted(RUN.glob('decision_*.json')) + sorted(RUN.glob('fold_*.json'))
    files += [RUN/'summary.json', RUN/'started.json']
    hashes = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['output_sha256'][path.name]:
            raise ValueError(f'Manifest hash mismatch: {path.name}')
        hashes[path.name] = digest
    decisions = [read(f) for f in sorted(RUN.glob('decision_*.json'))]
    folds = [read(f) for f in sorted(RUN.glob('fold_*.json'))]
    if len(decisions) != summary['decision_dates'] or len(folds) != summary['folds']:
        raise ValueError('Missing decision or fold records')
    normal = [d for d in decisions if not d['initial_construction_exempt']]
    tolerance = 1e-6
    for d in decisions:
        if (d['solver_status'] != 'optimal' or d['fallback_used'] or not d['passed']
                or d['maximum_stock_active'] > .015+tolerance
                or d['maximum_industry_active'] > .02+tolerance
                or d['maximum_style_active'] > .30+tolerance
                or d['target_weight_sum_error'] > tolerance or d['prohibited_buy_weight'] > tolerance
                or not d['eligibility_asof'] < d['signal_date'] < d['execution_date']):
            raise ValueError(f'Decision audit failed: {d["signal_date"]}')
        if not d['initial_construction_exempt'] and d['target_active_turnover'] > .15+tolerance:
            raise ValueError('Non-opening active turnover exceeded cap')
    if sum(d['initial_construction_exempt'] for d in decisions) != 1 or not decisions[0]['initial_construction_exempt']:
        raise ValueError('Unexpected opening exemption count')
    for f in folds:
        if f['last_training_label_end'] > f['model_cutoff']:
            raise ValueError('Training label cutoff failed')
        if f['inference_rows'] != 500*f['inference_dates']:
            raise ValueError('Unexpected aggregate inference universe count')
        c = f['calibration']
        if c and (c['last_label_end'] > f['model_cutoff'][:10] or c['history_end'] > f['model_cutoff'][:10]):
            raise ValueError('Calibration maturity failed')
        if f['model_parameters'] != folds[0]['model_parameters']:
            raise ValueError('Model parameters changed across folds')
    if not math.isclose(p['annualized_excess_return'], p['information_ratio']*p['annualized_tracking_error'],abs_tol=1e-12):
        raise ValueError('IR*TE identity failed')
    if p['newey_west_t_minimum'] != min(p['newey_west_t_by_lag'].values()):
        raise ValueError('NW decision lag selection changed')
    if not math.isclose(p['portfolio_cumulative_return']-p['benchmark_cumulative_return'],p['cumulative_excess_return_difference'],abs_tol=1e-12):
        raise ValueError('Cumulative return difference failed')
    oos = [f for f in folds if f['prediction_year'] >= 2019]
    result = {
        'source_commit': SOURCE_COMMIT, 'run_id':'d11_v1',
        'scope':'committed JSON hashes, audit records, and summary arithmetic; no daily NAV replication',
        'verified_json_outputs':len(files), 'manifest_status':manifest['status'],
        'run_code_commit':manifest['preflight']['repository']['head'],
        'historical_decision':'No-Go', 'new_oos_validation':False,
        'performance':p, 'previous_performance':old,
        'deltas':{k:p[k]-old[k] for k in ['annualized_excess_return','annualized_tracking_error','information_ratio','newey_west_t_minimum','maximum_relative_drawdown']},
        'reference_gates':{
            'annualized_excess_return':p['annualized_excess_return']>=.035,
            'information_ratio':p['information_ratio']>=.75,
            'newey_west_t':p['newey_west_t_minimum']>=2,
            'maximum_relative_drawdown':p['maximum_relative_drawdown']<=.10,
            'specific_contribution_share':None, 'doubled_impact_annualized_excess':None},
        'decision_audit':{
            'count':len(decisions),'optimal_count':len(decisions),'fallback_count':0,
            'opening_exemptions':1, 'normal_decisions':len(normal),
            'normal_at_turnover_cap':sum(abs(d['target_active_turnover']-.15)<=tolerance for d in normal),
            'normal_mean_target_active_turnover':mean(d['target_active_turnover'] for d in normal),
            'normal_mean_target_total_turnover':mean(d['target_total_turnover'] for d in normal),
            'maximum_stock_active':max(d['maximum_stock_active'] for d in decisions),
            'maximum_industry_active':max(d['maximum_industry_active'] for d in decisions),
            'maximum_style_active':max(d['maximum_style_active'] for d in decisions),
            'maximum_weight_sum_error':max(d['target_weight_sum_error'] for d in decisions),
            'maximum_prohibited_buy_weight':max(d['prohibited_buy_weight'] for d in decisions),
            'normal_median_pretrade_cash_weight':median(d['pretrade_cash_weight'] for d in normal),
            'maximum_actual_holdings_identifiers':max(d['actual_holdings'] for d in decisions)},
        'inference_audit':{
            'oos_dates':sum(f['inference_dates'] for f in oos),
            'oos_rows':sum(f['inference_rows'] for f in oos),
            'feature_abstentions':sum(f['feature_abstentions'] for f in oos),
            'rows_without_evaluation_label':sum(f['rows_without_evaluation_label'] for f in oos)},
        'folds':[ {k:f[k] for k in ['prediction_year','training_rows','inference_rows','inference_dates','feature_abstentions','rows_without_evaluation_label','model_cutoff','last_training_label_end','calibration']} for f in folds],
        'json_sha256':hashes,
        'not_recomputed':['daily/monthly returns','annual performance and Rank IC','new attribution','impact2x','full output verification'],
    }
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-summary',action='store_true')
    args=parser.parse_args()
    result=review()
    if args.write_summary:
        path=ROOT/'reports/d11_result_review.json'
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['json_sha256','folds','previous_performance']},ensure_ascii=False,indent=2))

"""Read-only verification of the D11 evidence bundle; never runs a strategy."""
from pathlib import Path
import csv
import hashlib
import json
import math

root = Path(__file__).resolve().parent
with (root / 'D11_sources_sha256.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    path = root / row['local_path']
    data = path.read_bytes()
    assert len(data) == int(row['bytes']), path
    assert hashlib.sha256(data).hexdigest() == row['sha256'], path
    if row['git_blob_sha1']:
        blob = b'blob ' + str(len(data)).encode() + b'\0' + data
        assert hashlib.sha1(blob).hexdigest() == row['git_blob_sha1'], path

def load(path):
    return json.loads((root / 'sources' / path).read_text(encoding='utf-8'))

s = load('reports/d9_v2_summary.json')
d = load('reports/d10_summary.json')
m = load('reports/d9_v2_manifest.json')
for key in ('original_manifest', 'recovery_manifest', 'feasibility_audit'):
    data = (root / 'sources' / m[key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m[key + '_sha256'], key
assert s['performance'] == d['frozen_d9_v2_performance']
p = s['performance']
assert math.isclose(p['annualized_excess_return'], p['information_ratio'] * p['annualized_tracking_error'], abs_tol=1e-12)
c = d['attribution']['oos']['annualized_contributions']
assert math.isclose(sum(c[k] for k in ('industry', 'style', 'specific_net', 'benchmark_bridge', 'unmodelled_bridge')), p['annualized_excess_return'], abs_tol=1e-12)
assert math.isclose(c['specific_net'] / p['annualized_excess_return'], d['attribution']['oos']['specific_share_net'], abs_tol=1e-12)
j = d['july_2026']
assert math.isclose(j['portfolio_return'] - j['benchmark_return'], j['excess_return'], abs_tol=1e-12)
assert d['decision'] == 'No-Go' and not d['diagnostic_acceptance']['passed']
checks = s['threshold_evaluation']['checks'].copy()
checks['specific_contribution_share'] = d['attribution']['oos']['specific_share_net'] >= 0.5
assert sum(v is True for v in checks.values()) == 2
assert sum(v is False for v in checks.values()) == 4
print(f'PASS: {len(rows)} source hashes, manifest links, main metrics, attribution, and six-gate reconciliation.')
print('Scope: report evidence only. No model training, optimization, backtest, or raw-data replication was performed.')

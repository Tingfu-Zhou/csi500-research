"""D12 P0 recheck of the D11 monthly table (第1天 下午 step 5).

Recomputes the headline figures from reports/d11/d11_diagnostic_v1/monthly_performance.tsv
using only the standard library, so the audit does not inherit the pandas/numpy
path that produced them. Anything that needs the daily parquet files is reported
as blocked rather than approximated.

    python3 scripts/d12_p0_monthly_recheck.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "reports/d11/d11_diagnostic_v1"
TOL = 1e-12


def read_tsv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def newey_west_t(values, lag):
    """t statistic of the mean under a Bartlett kernel with `lag` lags."""
    count = len(values)
    mean = sum(values) / count
    centred = [v - mean for v in values]
    variance = sum(v * v for v in centred) / count
    for step in range(1, lag + 1):
        cov = sum(centred[t] * centred[t - step] for t in range(step, count)) / count
        variance += 2 * (1 - step / (lag + 1)) * cov
    return mean / math.sqrt(variance / count)


def main():
    monthly = read_tsv(DIAG / "monthly_performance.tsv")
    saved = json.loads((DIAG / "summary.json").read_text())
    net = saved["performance"]["net"]
    excess = [float(row["excess_return"]) for row in monthly]
    count = len(excess)
    mean = sum(excess) / count
    sd = math.sqrt(sum((v - mean) ** 2 for v in excess) / (count - 1))

    annualized, tracking = mean * 12, sd * math.sqrt(12)
    checks = [("months", float(count), float(net["months"])),
              ("annualized_excess_return", annualized, net["annualized_excess_return"]),
              ("annualized_tracking_error", tracking, net["annualized_tracking_error"]),
              ("information_ratio", annualized / tracking, net["information_ratio"])]
    checks += [(f"newey_west_t_lag_{lag}", newey_west_t(excess, lag), net["newey_west_t_by_lag"][str(lag)])
               for lag in (3, 4, 5, 6)]
    checks.append(("cost_drag_annualized",
                   sum(float(r["cost_drag"]) for r in monthly) / count * 12,
                   saved["performance"]["cost_drag_annualized"]))
    checks.append(("same_path_gross_annualized",
                   sum(float(r["same_path_gross_excess"]) for r in monthly) / count * 12,
                   saved["performance"]["same_path_gross"]["annualized_excess_return"]))

    print(f"== recomputed from monthly_performance.tsv ({count} months, "
          f"{monthly[0]['month']} .. {monthly[-1]['month']}) ==")
    failures = []
    for name, got, want in checks:
        ok = abs(got - want) <= TOL
        failures += [] if ok else [name]
        print(f"  {name:30s} recomputed={got:< 20.14f} saved={want:< 20.14f} {'OK' if ok else 'DIFF'}")

    identity = [r["month"] for r in monthly
                if abs(float(r["portfolio_return"]) - float(r["benchmark_return"])
                       - float(r["excess_return"])) > 1e-15]
    gross = [r["month"] for r in monthly
             if abs(float(r["same_path_gross_excess"]) - float(r["excess_return"])
                    - float(r["cost_drag"])) > TOL]
    print(f"\n  excess == portfolio - benchmark, every month: {not identity}")
    print(f"  gross  == excess + cost_drag, every month:    {not gross}")

    order = sorted(range(count), key=lambda i: excess[i])
    print("\n== five weakest months by excess return ==")
    for i in order[:5]:
        print(f"  {monthly[i]['month']}  excess={excess[i]:+.6f}  trade_days={monthly[i]['trade_days']}")

    attribution = read_tsv(DIAG / "attribution_monthly.tsv")
    residual = sorted(attribution, key=lambda r: -abs(float(r["three_block_residual"])))
    print("\n== three months with the largest |three_block_residual| ==")
    print("   (attribution-quality outliers; the D11 gate for them is already 'failed')")
    for row in residual[:3]:
        rank = order.index(next(i for i, r in enumerate(monthly) if r["month"] == row["month"])) + 1
        print(f"  {row['month']}  residual={float(row['three_block_residual']):+.6f}  "
              f"excess={float(row['excess_return']):+.6f}  excess_rank={rank}/{count} (1 = weakest)")

    print("\n== blocked here: needs files absent from this checkout ==")
    for item, needs in (("maximum_relative_drawdown", "attribution_daily.parquet / baseline_replay_monthly.parquet"),
                        ("specific share and attribution gate", "attribution_daily.parquet"),
                        ("security-level drill-down of any month", "orders_*.parquet, trades.parquet, targets.parquet")):
        print(f"  {item:38s} needs {needs}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

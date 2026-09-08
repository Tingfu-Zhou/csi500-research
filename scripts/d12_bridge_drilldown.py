"""Day 2: take the benchmark bridge from month to day to security.

The bridge is the gap between the reconstructed constituent basket and the official
index. Over the 92 saved months its mean absolute size is 9.5x larger in June and
December than in the other ten, and 10 of its 12 largest months are June or December —
the CSI 500's reconstitution months. This script tests that reading against the data
instead of arguing it:

  stage 1  month -> day.  Where in the month does the bridge actually sit, and is that
           day a day the constituent set changed? Needs no Qlib provider.
  stage 2  day -> security.  Rebuild `sum_i wb_i * r_i` for the worst days, check it
           reproduces the saved bridge, then rank the basket contributions. Needs Qlib.

Stage 2 refuses to present a decomposition it cannot first reproduce: if the rebuilt
basket return does not match the saved one to `--tolerance`, it says so and reports the
gap rather than a security ranking. The official index stays one number throughout — the
plan forbids inventing a security-level difference against it.

    python scripts/d12_bridge_drilldown.py                        # stage 1 only
    python scripts/d12_bridge_drilldown.py --securities           # both stages
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_bridge import (basket_contributions, bridge_by_day, membership_timeline,
                            month_share_of_bridge, summarise_bridge)
from src.d12_repo import D12Error

#: The three months the plan names, which are the top three by |bridge| and, because
#: `unmodelled_bridge` and `accounting_residual` are ~1e-16, also the top three by
#: |three_block_residual|. The two columns agree to 1.9e-11 over all 92 months.
PLAN_MONTHS = ("2026-06", "2025-02", "2021-06")


def load_inputs(run, diagnostic, config):
    attribution = ROOT / diagnostic / "attribution_daily.parquet"
    weights = ROOT / Path(config["d9"]["v2"]["processed_oos_dir"]) / "benchmark_weights.parquet"
    for path in (attribution, weights):
        if not path.is_file():
            raise D12Error(f"missing {path}; it is a gitignored local artefact, not a repo file")
    return pd.read_parquet(attribution), pd.read_parquet(weights)


def security_stage(dates, exposures_path, config, lookback):
    """Rebuild the basket for `dates` and decompose it, or explain why it could not be."""
    from src.data_access import ResearchDataAccess
    from scripts.d3_pipeline import load_qlib_market

    exposures = pd.read_parquet(exposures_path, columns=["trade_date", "instrument",
                                                         "benchmark_weight"])
    exposures["trade_date"] = pd.to_datetime(exposures["trade_date"]).dt.normalize()
    access = ResearchDataAccess(ROOT)
    results = {}
    for trade_date, exposure_date in dates:
        weights = (exposures.loc[exposures["trade_date"].eq(exposure_date)]
                   .set_index("instrument")["benchmark_weight"].astype(float))
        if weights.empty:
            results[str(trade_date.date())] = {"error": f"no exposures at {exposure_date.date()}"}
            continue
        # Enough lead-in for the forward fill `d11_price_panels` applies; the reproduction
        # check below is what proves the window was long enough.
        start = trade_date - pd.Timedelta(days=lookback)
        market = load_qlib_market(access, sorted(weights.index), start, trade_date,
                                  float(config["tradability"]["qlib_amount_multiplier_to_cny"]))
        close = market.pivot(index="trade_date", columns="instrument", values="close").sort_index()
        close = close.where(close.gt(0) & np.isfinite(close)).ffill()
        if trade_date not in close.index or exposure_date not in close.index:
            results[str(trade_date.date())] = {"error": "market panel lacks the day or its predecessor"}
            continue
        returns = close.loc[trade_date] / close.loc[exposure_date] - 1
        table, summary = basket_contributions(weights, returns.reindex(weights.index))
        results[str(trade_date.date())] = {"exposure_date": str(exposure_date.date()), **summary,
                                           "table": table}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="*", default=list(PLAN_MONTHS),
                        help="months to drill into; empty means every month")
    parser.add_argument("--top-days", type=int, default=5)
    parser.add_argument("--securities", action="store_true",
                        help="also run stage 2; needs the Qlib provider")
    parser.add_argument("--top-securities", type=int, default=25)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--run", default="reports/d11/d11_v1")
    parser.add_argument("--diagnostic", default="reports/d11/d11_diagnostic_v1")
    parser.add_argument("--out-dir", default="reports/d12")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    attribution, weights = load_inputs(args.run, args.diagnostic, config)
    timeline = membership_timeline(weights)

    everything = bridge_by_day(attribution, timeline)
    summary = {"scope": "full period", **summarise_bridge(everything)}
    months = args.months or None
    focus = bridge_by_day(attribution, timeline, months=months)
    per_month = month_share_of_bridge(focus, top=args.top_days)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    focus.to_csv(out / "bridge_by_day.tsv", sep="\t", index=False)
    per_month.to_csv(out / "bridge_by_month.tsv", sep="\t", index=False)

    print(f"full period: {summary['days']} days, "
          f"total |bridge| = {summary['absolute_bridge_total'] * 100:.4f} pp, "
          f"signed = {summary['signed_bridge_total'] * 100:+.4f} pp")
    rebalance = summary["rebalance_months"]
    print(f"  June/December: {rebalance['share_of_days']:.1%} of days carry "
          f"{rebalance['share_of_absolute_bridge']:.1%} of the absolute bridge "
          f"({rebalance['mean_absolute_bridge_inside'] / rebalance['mean_absolute_bridge_outside']:.1f}x "
          f"the daily mean elsewhere)")
    if "membership_change_days" in summary:
        change = summary["membership_change_days"]
        print(f"  constituent changes: {change['days']} days ({change['share_of_days']:.1%}) carry "
              f"{change['share_of_absolute_bridge']:.1%} of the absolute bridge")
    for bucket, item in summary.get("by_snapshot_staleness", {}).items():
        print(f"  snapshot age {bucket:>5} trade days: {item['days']:5d} days, "
              f"mean |bridge| {item['mean_absolute_bridge'] * 1e4:7.3f} bp")

    print(f"\nper month (top {args.top_days} days):")
    for row in per_month.to_dict("records"):
        print(f"  {row['month']}  signed {row['signed_bridge'] * 100:+7.4f} pp  "
              f"|total| {row['absolute_bridge'] * 100:7.4f} pp  "
              f"top{args.top_days} = {row[f'share_in_top_{args.top_days}_days']:.1%}  "
              f"worst {row['worst_day']} {row['worst_day_bridge'] * 100:+.4f} pp "
              f"(membership changed: {row['worst_day_membership_changed']})")

    print(f"\nworst {args.top_days} days in scope:")
    worst = focus.sort_values("abs_bridge", ascending=False).head(args.top_days)
    for row in worst.to_dict("records"):
        print(f"  {row['trade_date'].date()}  bridge {row['benchmark_bridge'] * 100:+8.4f} pp  "
              f"weights from {row['exposure_date'].date()}  "
              f"members {row.get('members')}  +{row.get('added')}/-{row.get('removed')}  "
              f"L1 change {row.get('weight_l1_change', float('nan')):.4f}  "
              f"snapshot age {row.get('days_since_snapshot')}")

    payload = {"full_period": summary, "months_in_scope": sorted(focus["month"].unique()),
               "per_month": per_month.to_dict("records"),
               "worst_days": [{k: (str(v) if isinstance(v, pd.Timestamp) else v)
                               for k, v in row.items()} for row in worst.to_dict("records")]}

    if args.securities:
        pairs = [(row["trade_date"], row["exposure_date"]) for row in worst.to_dict("records")]
        stage2 = security_stage(pairs, ROOT / args.run / "factor_exposures.parquet",
                                config, args.lookback_days)
        saved = focus.set_index("trade_date")
        payload["securities"] = {}
        print("\nstage 2: rebuilding the basket per day")
        for key, item in stage2.items():
            if "error" in item:
                print(f"  {key}: {item['error']}")
                payload["securities"][key] = item
                continue
            table = item.pop("table")
            day = pd.Timestamp(key)
            expected = float(saved.loc[day, "benchmark_bridge"] + saved.loc[day, "benchmark_return"])
            item["saved_basket_return"] = expected
            item["reproduction_error"] = item["basket_return"] - expected
            item["reproduced"] = bool(abs(item["reproduction_error"]) <= args.tolerance)
            print(f"  {key}: rebuilt basket {item['basket_return']:+.8f} vs saved "
                  f"{expected:+.8f}  error {item['reproduction_error']:+.2e}  "
                  f"{'REPRODUCED' if item['reproduced'] else 'NOT reproduced'}")
            if item["reproduced"]:
                ranked = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
                ranked.head(args.top_securities).to_csv(
                    out / f"bridge_securities_{key.replace('-', '')}.tsv", sep="\t")
                item["top_contributions"] = ranked.head(args.top_securities)["contribution"].to_dict()
            else:
                print("       decomposition withheld: widen --lookback-days, or the saved run "
                      "used a different weight vector than factor_exposures.parquet")
            names = item.get("unobservable_names", [])
            if len(names) > 50:  # a long list belongs in the TSV, not the summary
                item["unobservable_names"] = names[:50] + [f"... {len(names) - 50} more"]
            payload["securities"][key] = item

    (out / "bridge_drilldown.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/bridge_by_day.tsv, bridge_by_month.tsv, bridge_drilldown.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

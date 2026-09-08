"""Re-derive D11's charged costs from the saved fills, and calibrate R1's coefficient.

Day 1 has to establish two things about the cost function before R1 may be specified:
that `src.d3_core.trade_cost` really is what the saved run paid, and how large the impact
coefficient actually was, so the R1 penalty is registered against observed magnitudes
rather than a guess.

`reports/d11/d11_v1/trades.parquet` is enough for both. It stores, per fill, the notional,
the four fee components and `impact_rate`, so every identity below is checked against the
run itself — no Qlib provider, no market frame, no re-simulation:

    impact_cost      == notional * impact_rate                        (execution ledger)
    impact_rate      == c * sqrt(dw),   dw = notional / signal_nav    (so c is recoverable)
    impact_cost/NAV  == c * dw ** 1.5                                 (R1's decision term)

The third line is the R1 objective term evaluated on D11's own trades. `c` recovered here
is `k * sigma * sqrt(NAV / A)` with the *realised* sigma and turnover; R1 must estimate it
at T instead, and the quantiles printed here are what that estimate has to land near.

    python3 scripts/d12_impact_reconciliation.py
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

from src.d3_core import CostModel

TRADE_COLUMNS = ["execution_date", "signal_nav", "executed_trade_amount", "commission_cost",
                 "transfer_fee_cost", "stamp_duty_cost", "impact_cost", "total_cost", "impact_rate"]
QUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]


def _deviation(observed, expected):
    """Max absolute and relative gap, ignoring rows where both sides are zero."""
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    absolute = np.abs(observed - expected)
    scale = np.maximum(np.abs(expected), 1.0)
    return {"max_absolute": float(absolute.max()), "max_relative": float((absolute / scale).max())}


def reconcile(trades, model, saved_totals=None):
    """Check every cost identity on the saved fills and recover the impact coefficient."""
    missing = [column for column in TRADE_COLUMNS if column not in trades.columns]
    if missing:
        raise KeyError(f"trades frame is missing {missing}")
    frame = trades.loc[:, TRADE_COLUMNS].copy()
    frame["execution_date"] = pd.to_datetime(frame["execution_date"])
    notional = frame["executed_trade_amount"].abs()
    sells = frame["executed_trade_amount"] < 0
    stamp = frame["execution_date"].map(model.stamp_rate)

    identities = {
        "impact_cost == notional * impact_rate":
            _deviation(frame["impact_cost"], notional * frame["impact_rate"]),
        "commission_cost == notional * commission_rate":
            _deviation(frame["commission_cost"], notional * model.commission_rate),
        "transfer_fee_cost == notional * transfer_fee_rate":
            _deviation(frame["transfer_fee_cost"], notional * model.transfer_fee_rate),
        "stamp_duty_cost == notional * stamp_rate on sells only":
            _deviation(frame["stamp_duty_cost"], notional * stamp * sells),
        "total_cost == sum of the four components":
            _deviation(frame["total_cost"], frame[["commission_cost", "transfer_fee_cost",
                                                   "stamp_duty_cost", "impact_cost"]].sum(axis=1)),
    }

    # Recover c from the ledger: impact_rate = c * sqrt(dw), so c = impact_rate / sqrt(dw).
    traded = notional.gt(0) & frame["signal_nav"].gt(0)
    weight = (notional / frame["signal_nav"]).where(traded)
    coefficient = (frame["impact_rate"] / np.sqrt(weight)).where(traded)
    identities["impact_cost / NAV == c * |dw| ** 1.5 (the R1 term)"] = _deviation(
        (frame["impact_cost"] / frame["signal_nav"]).where(traded, 0.0).fillna(0.0),
        (coefficient * weight ** 1.5).where(traded, 0.0).fillna(0.0))

    usable = coefficient.replace([np.inf, -np.inf], np.nan).dropna()
    totals = {name: float(frame[name].sum()) for name in
              ["commission_cost", "transfer_fee_cost", "stamp_duty_cost", "impact_cost", "total_cost"]}
    result = {
        "rows": int(len(frame)), "filled_rows": int(traded.sum()),
        "impact_coefficient_config": float(model.impact_coefficient),
        "totals_cny": totals,
        "recovered_coefficient_c": {
            "definition": "c = k * sigma * sqrt(NAV / A), realised; R1 must estimate this at T",
            "count": int(len(usable)),
            "quantiles": {str(q): float(usable.quantile(q)) for q in QUANTILES},
            "mean": float(usable.mean())},
        "traded_weight_dw": {
            "quantiles": {str(q): float(weight.dropna().quantile(q)) for q in QUANTILES},
            "max": float(weight.max())},
        "expected_impact_of_a_one_percent_trade_bps": {
            str(q): float(usable.quantile(q) * 0.01 ** 1.5 * 1e4) for q in QUANTILES},
        "identities": identities,
    }
    if saved_totals:
        result["against_saved_summary"] = {
            name: {"recomputed": totals[name], "saved": float(value),
                   "difference": totals[name] - float(value)}
            for name, value in saved_totals.items() if name in totals}
    worst = max(item["max_relative"] for item in identities.values())
    result["identities_max_relative_deviation"] = worst
    result["identities_hold"] = bool(worst <= 1e-9)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="reports/d11/d11_v1")
    parser.add_argument("--diagnostic", default="reports/d11/d11_diagnostic_v1")
    parser.add_argument("--out", default="reports/d12/impact_reconciliation.json")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    trades_path = ROOT / args.run / "trades.parquet"
    if not trades_path.is_file():
        print(f"missing {trades_path}\n"
              "It is a saved-run output kept out of Git by .gitignore; run this on the machine "
              "that produced d11_v1.", file=sys.stderr)
        return 2

    saved_totals = {}
    summary_path = ROOT / args.diagnostic / "summary.json"
    if summary_path.is_file():
        execution = json.loads(summary_path.read_text())["execution"]
        saved_totals = {**execution["costs_cny"], "total_cost": execution["total_cost_cny"]}

    result = reconcile(pd.read_parquet(trades_path), CostModel.from_config(config), saved_totals)
    result["source_run"] = args.run
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}\n{result['rows']} order rows, {result['filled_rows']} with a fill")
    for name, deviation in result["identities"].items():
        print(f"  {'OK ' if deviation['max_relative'] <= 1e-9 else 'DIFF'} {name}")
        print(f"       max |diff| = {deviation['max_absolute']:.3e} cny, "
              f"max relative = {deviation['max_relative']:.3e}")
    print("\nrecovered impact coefficient c = k * sigma * sqrt(NAV / A):")
    for quantile, value in result["recovered_coefficient_c"]["quantiles"].items():
        cost = result["expected_impact_of_a_one_percent_trade_bps"][quantile]
        print(f"  p{float(quantile) * 100:>5.1f}  c = {value:10.4f}   "
              f"a 1% trade would expect {cost:7.2f} bp of NAV")
    for name, item in result.get("against_saved_summary", {}).items():
        print(f"  total {name:<20} recomputed {item['recomputed']:16.4f}  "
              f"saved {item['saved']:16.4f}  diff {item['difference']:+.3e}")
    return 0 if result["identities_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

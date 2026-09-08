"""How much slower is R1's objective than R0's? Timed on synthetic data, no Qlib needed.

Day 1 has to bound the R1 full-period runtime, and the two saved runs recorded no wall
clock at all, so there is nothing to extrapolate from. This probe rebuilds the constraint
set `src.d11_core.optimize_at_signal` solves — budget, long-only, +-1.5% stock, +-2%
industry, +-0.3 style, 15% single-sided turnover, linear fees — at the real problem size,
then times it with and without R1's expected-impact penalty.

The numbers are a shape, not a forecast: synthetic mu and exposures, this machine's cores,
no data loading, no accounting, no attribution. What it does answer is whether the power
cone CLARABEL needs for `|dw| ** 1.5` costs a factor of two or a factor of fifty.

    python3 scripts/d12_solver_probe.py --repeats 5
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d12_core import ImpactEstimator, expected_impact_fraction, impact_penalty

#: Observed in reports/d11/d11_v1/decision_*.json: 392 decisions, NAV 1.0e7 -> 3.1e7,
#: 472 holdings at the median, and every regular decision sitting on the turnover cap.
DECISIONS, MEDIAN_NAV = 392, 18_797_377.0


def problem_inputs(names, industries, seed):
    rng = np.random.default_rng(seed)
    benchmark = rng.uniform(0.5, 1.5, names)
    benchmark /= benchmark.sum()
    membership = np.zeros((industries, names))
    membership[rng.integers(0, industries, names), np.arange(names)] = 1.0
    styles = rng.normal(size=(2, names))
    return {"benchmark": benchmark, "pretrade": benchmark.copy(),
            "mu": rng.normal(0, 0.01, names), "industry": membership, "style": styles,
            # c = k * sigma * sqrt(NAV / A) at the registered k, with plausible sigma and turnover.
            "coefficient": 0.5 * rng.uniform(0.01, 0.05, names)
            * np.sqrt(MEDIAN_NAV / rng.uniform(3e7, 2e9, names))}


def solve_once(inputs, config, *, penalised, mode="split"):
    import cvxpy as cp

    limits, costs = config["d7"], config["costs"]
    names = len(inputs["mu"])
    w = cp.Variable(names)
    active = w - inputs["benchmark"]
    delta = w - inputs["pretrade"]
    buys, sells = cp.sum(cp.pos(delta)), cp.sum(cp.pos(-delta))
    fee = (costs["commission_rate"] + costs["transfer_fee_rate"]) * (buys + sells)
    fee += costs["stamp_duty"][-1]["rate"] * sells
    objective = inputs["mu"] @ active - fee
    if penalised:
        objective = objective - impact_penalty(w, inputs["pretrade"], inputs["coefficient"], mode=mode)
    constraints = [cp.sum(w) == 1, w >= 0,
                   active <= limits["stock_active_limit"], active >= -limits["stock_active_limit"],
                   inputs["industry"] @ active <= limits["industry_active_limit"],
                   inputs["industry"] @ active >= -limits["industry_active_limit"],
                   inputs["style"] @ active <= limits["style_active_limit"],
                   inputs["style"] @ active >= -limits["style_active_limit"],
                   0.5 * cp.norm1(delta) <= limits["single_sided_turnover_limit"]]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    started = time.perf_counter()
    problem.solve(solver=config["d7"]["solver"])
    elapsed = time.perf_counter() - started
    # Status is recorded, not raised on. `optimize_at_signal` accepts "optimal" alone, so
    # how often a variant lands on "optimal_inaccurate" is one of the things to measure.
    sample = {"seconds": elapsed, "status": problem.status, "target": None,
              "active_turnover": None, "expected_impact": None}
    if w.value is None:
        return sample
    target = np.maximum(w.value, 0.0)
    sample.update(target=target,
                  active_turnover=float(0.5 * np.abs(target - inputs["pretrade"]).sum()),
                  expected_impact=expected_impact_fraction(
                      pd.Series(target - inputs["pretrade"]), pd.Series(inputs["coefficient"])))
    return sample


def summarise(samples):
    seconds = [item["seconds"] for item in samples]
    statuses = {}
    for item in samples:
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    solved = [item for item in samples if item["target"] is not None]
    return {"repeats": len(seconds), "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds), "max_seconds": max(seconds),
            "status_counts": statuses,
            "strictly_optimal": statuses.get("optimal", 0),
            "median_active_turnover":
                statistics.median(item["active_turnover"] for item in solved) if solved else None,
            "median_expected_impact_bp_of_nav":
                statistics.median(item["expected_impact"] for item in solved) * 1e4 if solved else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=int, default=500, help="universe size; CSI500 is ~500")
    parser.add_argument("--industries", type=int, default=31, help="SW2021 level-1 count")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decisions", type=int, default=DECISIONS)
    parser.add_argument("--out", default="reports/d12/solver_probe.json")
    args = parser.parse_args()

    import cvxpy as cp
    import yaml

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    estimator = ImpactEstimator.from_config(config)
    assert estimator.k == config["costs"]["impact"]["coefficient"]

    variants = {"r0_linear": dict(penalised=False),
                "r1_penalised_split": dict(penalised=True, mode="split"),
                "r1_penalised_abs": dict(penalised=True, mode="abs")}
    results, targets = {}, {}
    for label, options in variants.items():
        samples = [solve_once(problem_inputs(args.names, args.industries, seed), config, **options)
                   for seed in range(args.repeats)]
        results[label] = summarise(samples)
        targets[label] = [item["target"] for item in samples]

    base = results["r0_linear"]["median_seconds"]
    for label, item in results.items():
        item["slowdown_vs_r0"] = item["median_seconds"] / base
        item["projected_full_period_minutes"] = item["median_seconds"] * args.decisions / 60
    # The two penalty spellings must be the same function, not merely similar.
    pairs = [(a, b) for a, b in zip(targets["r1_penalised_split"], targets["r1_penalised_abs"])
             if a is not None and b is not None]
    agreement = max((float(np.abs(a - b).max()) for a, b in pairs), default=None)

    payload = {"recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "machine": {"platform": platform.platform(), "python": platform.python_version(),
                           "cvxpy": cp.__version__, "solver": config["d7"]["solver"]},
               "problem": {"names": args.names, "industries": args.industries,
                           "decisions_projected": args.decisions, "nav": MEDIAN_NAV,
                           "note": "synthetic mu and exposures; solver time only, no data or accounting"},
               "results": results,
               "penalty_modes_agree_to": agreement,
               "caveat": "a shape for planning, not a runtime forecast for the real pipeline"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"{args.names} names, {args.industries} industries, {args.repeats} repeats, "
          f"solver={config['d7']['solver']}")
    for label, item in results.items():
        turnover = item["median_active_turnover"]
        print(f"  {label:<20} median {item['median_seconds']:7.3f}s  "
              f"x{item['slowdown_vs_r0']:5.2f} vs R0  "
              f"{args.decisions} decisions ~ {item['projected_full_period_minutes']:6.2f} min  "
              f"turnover {turnover:.4f}" if turnover is not None else
              f"  {label:<20} median {item['median_seconds']:7.3f}s  (no solution)")
        print(f"  {'':<20} solver status: {json.dumps(item['status_counts'])}")
    print(f"  split and abs penalties agree to {agreement:.3e} in weight"
          if agreement is not None else "  penalty modes not comparable: a solve returned nothing")
    strict = {label: item["strictly_optimal"] for label, item in results.items()}
    if any(count < args.repeats for count in strict.values()):
        print(f"\n  NOTE optimize_at_signal accepts status 'optimal' only and raises otherwise.\n"
              f"       strictly optimal out of {args.repeats}: {json.dumps(strict)}\n"
              f"       On synthetic data this is a conditioning artefact, not a verdict on the\n"
              f"       real problem, but R1 must record how the real solve behaves before it runs.")
    print("\nSolver time only. Data loading, matching, accounting, attribution and the two "
          "3.9 GB input hashes are extra; measure them with scripts/d12_preflight.py --hash-inputs.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

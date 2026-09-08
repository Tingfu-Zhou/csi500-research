"""R1's decision-side expected impact: the same cost D11 charges, priced before the fill.

`src.d3_core.trade_cost` charges, per fill,

    impact_cny = notional * k * sigma * sqrt(notional / A)

and with `notional = |dw| * NAV` that is, as a fraction of NAV,

    impact_fraction = k * sigma * sqrt(NAV / A) * |dw| ** 1.5          (*)

(*) is what R1 subtracts inside the optimiser. Same k, same sigma statistic, same units.
The single difference is *when* the inputs are read: execution charges against the T+1
quote, so at signal date T the amount has to be estimated and the volatility has to be
taken from the window that ends at T.

  sigma  `scripts.d3_pipeline.add_ex_ante_volatility` computes a 20-day rolling standard
         deviation of daily close returns and then shifts it one trade day, so the value
         charged at T+1 is the window ending at T. `rolling_estimates` computes the same
         statistic and does *not* shift it, so its row at T carries exactly what execution
         will charge at T+1 — read from the T panel, with no T+1 row touched. Shifting
         here as well would lag the estimate twice; reading the T+1 row would use a
         cross-section that is not yet observable at T.

  A      the realised T+1 turnover cannot be estimated without error. `rolling_estimates`
         uses the mean daily turnover over the 20 trade days ending at T, requiring 15
         valid observations, and never silently prices a missing estimate at zero: see
         `coefficients_at`.

Nothing here reads a label, a T+1 quote, or a D11 output. This module supplies the
estimate and the convex penalty term only; wiring them into a backtest is separate work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.d12_repo import D12Error

#: Columns `rolling_estimates` needs from the D11 market frame (`prepare_oos_market`).
MARKET_COLUMNS = ("trade_date", "instrument", "close", "daily_amount_cny")


@dataclass(frozen=True)
class ImpactEstimator:
    """Decision-side impact parameters. Registered before R1 runs; never fitted."""

    k: float
    amount_lookback: int = 20
    amount_minimum_observations: int = 15
    volatility_lookback: int = 20
    volatility_minimum_observations: int = 20
    fallback_daily_volatility: float = 0.03
    unavailable_amount_quantile: float = 0.90

    @classmethod
    def from_config(cls, config, *, k=None):
        """Reuse the execution-side numbers; `k` overrides only the decision-side one."""
        impact = config["costs"]["impact"]
        estimator = cls(
            k=float(impact["coefficient"] if k is None else k),
            amount_lookback=int(config["tradability"]["amount_lookback_trade_days"]),
            volatility_lookback=int(impact["volatility_lookback_trade_days"]),
            volatility_minimum_observations=int(impact["minimum_observations"]),
            fallback_daily_volatility=float(impact["fallback_daily_volatility"]))
        estimator.validate()
        return estimator

    def validate(self):
        if not math.isfinite(self.k) or self.k < 0:
            raise D12Error("decision-side impact coefficient must be finite and non-negative")
        if not 0 < self.amount_minimum_observations <= self.amount_lookback:
            raise D12Error("amount minimum observations must fit inside the lookback")
        if not 0 < self.volatility_minimum_observations <= self.volatility_lookback:
            raise D12Error("volatility minimum observations must fit inside the lookback")
        if not math.isfinite(self.fallback_daily_volatility) or self.fallback_daily_volatility < 0:
            raise D12Error("fallback volatility must be finite and non-negative")
        if not 0 <= self.unavailable_amount_quantile <= 1:
            raise D12Error("substitute quantile must lie in [0, 1]")

    @property
    def bypassed(self) -> bool:
        """k=0 is the R0 replay.

        The penalty is then dropped from the objective *structurally* — the problem stays
        the linear program D11 solved — rather than added as a zero-valued nonlinear term
        that would change the cone and perturb the solution. The missing-amount rules are
        bypassed with it, so a k=0 run cannot forbid a buy that R0 was allowed to make.
        """
        return self.k == 0.0


def rolling_estimates(market, estimator):
    """Per (trade_date, instrument) estimates that read nothing after `trade_date`.

    Returns one row per market row with `sigma_est`, `sigma_source`, `a_est` and
    `a_observations`. `sigma_source` records which of the three tiers supplied the value,
    so a run can report how often it fell back rather than burying it in an average.
    """
    missing = [column for column in MARKET_COLUMNS if column not in market.columns]
    if missing:
        raise D12Error(f"market frame is missing {missing}")
    frame = market.loc[:, list(MARKET_COLUMNS)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise D12Error("market frame has duplicate (trade_date, instrument) rows")
    frame = frame.sort_values(["instrument", "trade_date"]).reset_index(drop=True)

    grouped = frame.groupby("instrument", sort=False)
    frame["return_1d"] = grouped["close"].pct_change(fill_method=None)
    frame["sigma_rolling"] = frame.groupby("instrument", sort=False)["return_1d"].transform(
        lambda series: series.rolling(estimator.volatility_lookback,
                                      min_periods=estimator.volatility_minimum_observations).std())

    amount = pd.to_numeric(frame["daily_amount_cny"], errors="coerce")
    frame["amount_valid"] = amount.where(np.isfinite(amount) & amount.gt(0))
    by_instrument = frame.groupby("instrument", sort=False)["amount_valid"]
    frame["a_est"] = by_instrument.transform(
        lambda series: series.rolling(estimator.amount_lookback,
                                      min_periods=estimator.amount_minimum_observations).mean())
    frame["a_observations"] = by_instrument.transform(
        lambda series: series.rolling(estimator.amount_lookback, min_periods=1).count())

    # Same fallback ladder as the execution side: same-date cross-sectional median, then
    # the registered constant. Both tiers stay inside date T's own cross-section.
    median = frame.groupby("trade_date")["sigma_rolling"].transform("median")
    filled = frame["sigma_rolling"].fillna(median)
    frame["sigma_source"] = np.where(frame["sigma_rolling"].notna(), "rolling",
                                     np.where(filled.notna(), "cross_section_median",
                                              "constant_fallback"))
    frame["sigma_est"] = filled.fillna(estimator.fallback_daily_volatility)
    if frame["sigma_est"].isna().any() or frame["sigma_est"].lt(0).any():
        raise D12Error("ex-ante volatility estimate failed")

    columns = ["trade_date", "instrument", "sigma_est", "sigma_source", "a_est", "a_observations"]
    return frame.loc[:, columns].sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def coefficients_at(estimates, asof, nav, estimator, *, instruments=None):
    """Coefficients `c` with expected impact = ``sum_i c_i * |dw_i| ** 1.5``.

    Returns `(coefficients, buy_forbidden, diagnostics)`. Where the turnover estimate is
    unusable the plan's rule applies rather than a zero cost: no new buying in that name,
    and any sell priced at the 90th percentile of the coefficients that *are* usable in
    the same cross-section. With no usable cross-section at all the run stops.
    """
    asof = pd.Timestamp(asof).normalize()
    rows = estimates.loc[estimates["trade_date"].eq(asof)]
    if rows["instrument"].duplicated().any():
        raise D12Error(f"{asof.date()}: duplicate instruments in the estimate cross-section")
    rows = rows.set_index("instrument")
    if instruments is not None:
        rows = rows.reindex(pd.Index(instruments, name="instrument"))
    nav = float(nav)
    if not math.isfinite(nav) or nav <= 0:
        raise D12Error(f"{asof.date()}: NAV must be finite and positive")

    if estimator.bypassed:
        zero = pd.Series(0.0, index=rows.index, name="coefficient")
        allowed = pd.Series(False, index=rows.index, name="buy_forbidden")
        return zero, allowed, {"as_of": str(asof.date()), "instruments": int(len(rows)),
                               "bypassed_for_zero_k": True, "amount_unavailable": 0,
                               "substitute_coefficient": None, "sigma_source_counts": {}}

    sigma = pd.to_numeric(rows["sigma_est"], errors="coerce")
    amount = pd.to_numeric(rows["a_est"], errors="coerce")
    usable = (np.isfinite(amount) & amount.gt(0) & np.isfinite(sigma) & sigma.ge(0)).fillna(False)
    coefficient = estimator.k * sigma * np.sqrt(nav / amount.where(usable))

    if not usable.any():
        raise D12Error(f"{asof.date()}: no instrument has a usable turnover estimate; "
                       "R1 stops here rather than pricing impact at zero")
    substitute = float(coefficient.loc[usable].quantile(estimator.unavailable_amount_quantile))
    coefficient = coefficient.where(usable, substitute).astype(float)
    coefficient.name = "coefficient"
    buy_forbidden = (~usable).rename("buy_forbidden")
    sources = rows["sigma_source"].value_counts().to_dict() if "sigma_source" in rows else {}
    diagnostics = {"as_of": str(asof.date()), "instruments": int(len(rows)),
                   "bypassed_for_zero_k": False,
                   "amount_unavailable": int((~usable).sum()),
                   "substitute_coefficient": substitute,
                   "sigma_source_counts": {str(key): int(value) for key, value in sources.items()}}
    return coefficient, buy_forbidden, diagnostics


def impact_fraction_of_trade(weight_change, sigma, amount, nav, k):
    """Single-name form of (*), in NAV fraction. The scalar the equivalence test pins."""
    if amount <= 0 or not math.isfinite(amount):
        raise D12Error("turnover estimate must be finite and positive")
    return k * sigma * math.sqrt(nav / amount) * abs(weight_change) ** 1.5


def expected_impact_fraction(trade_weights, coefficients):
    """``sum_i c_i * |dw_i| ** 1.5``, as a fraction of NAV."""
    weights = pd.Series(trade_weights, dtype=float)
    coefficient = pd.Series(coefficients, dtype=float).reindex(weights.index)
    if coefficient.isna().any():
        raise D12Error("every traded instrument needs an impact coefficient")
    return float((coefficient * weights.abs() ** 1.5).sum())


def expected_impact_cny(trade_weights, coefficients, nav):
    """The same quantity in yuan, for reconciliation against the execution ledger."""
    return expected_impact_fraction(trade_weights, coefficients) * float(nav)


def impact_penalty(w, old, coefficients, *, mode="abs"):
    """The cvxpy penalty term, as a fraction of NAV.

    Two spellings of the same function. `mode="abs"` is the literal form of (*) and needs
    one power cone per name; `mode="split"` prices the buy and sell legs separately and
    needs two. They agree to ~3e-6 in weight, but they do not behave alike numerically:
    over 20 synthetic solves at the real problem size `scripts/d12_solver_probe.py` had
    CLARABEL return status "optimal" 20/20 times for `abs` and 6/20 for `split`, and `abs`
    was the faster of the two. `optimize_at_signal` accepts "optimal" and raises on
    anything else, so `abs` is the default and `split` is kept only for cross-checking.

    `|dw| ** 1.5` is convex, so the penalised problem stays convex — but it is no longer a
    linear program. Convex marginal cost may reduce corner solutions; it does not
    guarantee an interior one, and in the probe turnover still sat on the 15% cap.
    """
    import cvxpy as cp

    coefficient = np.asarray(coefficients, dtype=float)
    if coefficient.ndim != 1 or coefficient.shape[0] != w.shape[0]:
        raise D12Error("coefficient vector must match the optimisation variable")
    if not np.isfinite(coefficient).all() or (coefficient < 0).any():
        raise D12Error("impact coefficients must be finite and non-negative")
    delta = w - np.asarray(old, dtype=float)
    if mode == "split":
        magnitude = cp.power(cp.pos(delta), 1.5) + cp.power(cp.pos(-delta), 1.5)
    elif mode == "abs":
        magnitude = cp.power(cp.abs(delta), 1.5)
    else:
        raise D12Error(f"unknown penalty mode {mode!r}")
    return cp.sum(cp.multiply(coefficient, magnitude))

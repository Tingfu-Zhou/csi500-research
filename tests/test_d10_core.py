from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.d3_core import CostModel, match_target_orders, run_vwap_backtest, trade_cost
from src.d4_core import build_forward_excess_labels
from src.d8_core import summarize_backtest_performance
from src.d10_core import (DiagnosticError, align_stress_daily, attribute_executed_path,
    executed_turnover_split, half_l1, linked_monthly, path_gross_net, price_panels,
    reconstruct_executed_path, require_close, stress_schedule, tradable_fill_range,
    assert_long_only, turnover_split, wls_cross_section)
from src.d10_io import (D10Output, assert_snapshot_unchanged, git_blob, snapshot_inputs, verify_repository)


def synthetic_case(*, n=12, periods=12, model=None):
    rng = np.random.default_rng(37)
    dates = pd.bdate_range("2023-08-21", periods=periods)
    names = [f"S{i:02}" for i in range(n)]
    close = 10 * np.cumprod(1 + rng.normal(.001, .012, (periods, n)), axis=0)
    vwap = close / (1 + rng.normal(0, .006, (periods, n)))
    market = pd.DataFrame([{ "trade_date": d, "instrument": name, "close": close[t, i],
        "vwap": vwap[t, i], "daily_amount_cny": 10_000_000., "volatility": .02,
        "can_buy": True, "can_sell": True} for t, d in enumerate(dates) for i, name in enumerate(names)])
    targets = pd.DataFrame([{"signal_date": dates[t], "instrument": name,
        "target_weight": float(i % 2 == (t // 3) % 2) * 2 / n}
        for t in range(0, periods - 1, 3) for i, name in enumerate(names)])
    model = model or CostModel(.00015, .00001, ((pd.Timestamp("1900-01-01"), pd.Timestamp("2023-08-27"), .001),
                    (pd.Timestamp("2023-08-28"), pd.Timestamp("2100-01-01"), .0005)), .5, .05)
    result = run_vwap_backtest(market, targets, 10_000., model)
    benchmark = pd.DataFrame({"trade_date": dates, "close": np.mean(close, axis=1)})
    _, daily = summarize_backtest_performance(result.nav, benchmark, initial_capital=10_000.)
    x = rng.normal(size=(n, 2))
    exposures = pd.DataFrame([{"trade_date": d, "instrument": name,
        "industry_A": float(i < n // 2), "industry_B": float(i >= n // 2),
        "industry_absent": 0., "style_Size": x[i, 0], "style_Beta": x[i, 1],
        "market_cap": float(100 + i ** 2), "benchmark_weight": 1 / n}
        for d in dates for i, name in enumerate(names)])
    return market, targets, model, result, daily, exposures


def test_reconstruct_intraday_costs_and_cash_matches_frozen_d3():
    market, _, model, result, daily, _ = synthetic_case()
    path = reconstruct_executed_path(daily, result.trades, market, 10_000, model)
    assert path.daily.nav_error_cny.max() < 1e-8
    assert path.daily.accounting_residual.abs().max() < 1e-12
    assert abs(path.daily.execution_return.iloc[0]) > 1e-8
    assert path.daily.holding_return.iloc[0] == 0
    assert path.daily.cost_return.sum() > 0


def test_reconstruction_blocked_exit_keeps_units_and_no_fee():
    market, targets, model, _, _, _ = synthetic_case()
    second_execution = sorted(targets.signal_date.unique())[1] + pd.offsets.BDay()
    market.loc[market.trade_date.eq(second_execution), "can_sell"] = False
    result = run_vwap_backtest(market, targets, 10_000., model)
    benchmark = pd.DataFrame({"trade_date": sorted(market.trade_date.unique()), "close": 100.})
    _, daily = summarize_backtest_performance(result.nav, benchmark, initial_capital=10_000.)
    path = reconstruct_executed_path(daily, result.trades, market, 10_000., model)
    blocked = result.trades.loc[result.trades.blocked_by_tradability]
    assert len(blocked) > 0 and blocked.executed_trade_amount.eq(0).all() and blocked.total_cost.eq(0).all()
    assert path.ending_weights.loc[second_execution, blocked.instrument].sum() > 0


@pytest.mark.parametrize("column", ["nav", "cash", "portfolio_return", "transaction_cost", "single_sided_turnover"])
def test_corrupted_frozen_daily_fails_closed(column):
    market, _, model, result, daily, _ = synthetic_case()
    daily.loc[2, column] += 1.0
    with pytest.raises(DiagnosticError):
        reconstruct_executed_path(daily, result.trades, market, 10_000, model)


def test_corrupted_order_cost_is_detected_independently_of_nav():
    market, _, model, result, daily, _ = synthetic_case()
    result.trades.loc[0, "stamp_duty_cost"] += 1
    with pytest.raises(DiagnosticError, match="stamp_duty_cost"):
        reconstruct_executed_path(daily, result.trades, market, 10_000, model)


def test_long_only_contract_is_asserted_in_currency_not_share_count():
    """D3 refuses a position below -1e-8 CNY; a share count is not scale free.

    A 1.5 CNY stock held at 1.5m CNY carries 1e6 units, so the float64 residue
    of a full exit can pass -1e-10 UNITS while being worth -2e-10 CNY, which is
    43x inside D3's own contract. Bounding units rejects that ledger; bounding
    currency, as D3 states it, accepts exactly what D3 produced.
    """
    price, position = 1.5, 1_500_000.0
    units = position / price
    residue = -1.164e-10                       # observed scale of u - fl(fl(u*p)/p)
    assert abs(residue) > 1e-10                # the old share-count guard would have raised
    assert abs(residue * price) < 1e-8         # yet D3's currency contract is untouched
    assert_long_only(pd.Series({"CHEAP": residue * price, "HELD": position}), "2018-05-07", "pretrade position")
    with pytest.raises(DiagnosticError, match="long-only contract"):
        assert_long_only(pd.Series({"SHORTED": -1e-7}), "2018-05-07", "pretrade position")
    with pytest.raises(DiagnosticError, match="SHORTED=-12345"):
        assert_long_only(pd.Series({"OK": 5.0, "SHORTED": -12345.0}), "2018-05-07", "post-trade position")
    assert units > 4.5e5  # only this many units makes the residue reach the old threshold


def test_negative_dust_position_fills_through_the_sell_leg_not_a_forbidden_buy():
    """D3 gates legs, not the sign of the net fill.

    A full exit books back (units*price)/price, which in float64 can overshoot
    and leave a negative unit residue above the 1e-12 prune. On a later
    rebalance the sell leg is clipped UP TO that negative position, so -0.0
    becomes a positive, buy-direction fill of the dust that can_sell allowed.
    Reading the leg off the sign reports a forbidden buy that never happened.
    """
    model = CostModel(.00015, .00001, ((pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01"), .001)), .5, .05)
    names = ["BLOCKED", "OPEN"]
    quotes = pd.DataFrame({"execution_price": [10., 10.], "daily_amount_cny": [6e7, 6e7],
        "volatility": [.015, .015], "can_buy": [False, True], "can_sell": [True, True]}, index=names)
    dust = -1.403375e-11
    result = match_target_orders(pd.Series({"BLOCKED": dust, "OPEN": 100_000.}), 9_900_000.,
                                 pd.Series({"BLOCKED": .01, "OPEN": .01}), quotes,
                                 pd.Timestamp("2017-02-13"), model)
    order = result.trades.set_index("instrument").loc["BLOCKED"]
    assert order.executed_trade_amount == -dust > 0 and not order.can_buy and order.blocked_by_tradability
    low, high = tradable_fill_range(order.desired_trade_amount, dust, 6e7, False, True, .05)
    assert low == high == -dust
    # The same order against a clean zero position may not fill at all.
    assert tradable_fill_range(order.desired_trade_amount, 0., 6e7, False, True, .05) == (0., 0.)


def test_fill_range_still_refuses_forbidden_and_oversized_fills():
    assert tradable_fill_range(50_000., 0., 6e7, False, True, .05) == (0., 0.)
    assert tradable_fill_range(-50_000., 50_000., 6e7, True, False, .05) == (0., 0.)
    # a sell is pinned to one point; a buy may be scaled down by the cash sweep
    assert tradable_fill_range(-50_000., 50_000., 6e7, True, True, .05) == (-50_000., -50_000.)
    assert tradable_fill_range(50_000., 0., 6e7, True, True, .05) == (0., 50_000.)
    # participation caps the buy leg at 5% of the day's amount
    assert tradable_fill_range(5e6, 0., 6e7, True, True, .05) == (0., 3e6)
    # a sell can never exceed the position it is clipped to
    assert tradable_fill_range(-90_000., 40_000., 6e7, True, True, .05) == (-40_000., -40_000.)


def test_overfilled_order_beyond_the_leg_range_fails_closed():
    """Cost-consistent inflation: only the leg range can catch this one."""
    market, _, model, result, daily, _ = synthetic_case()
    traded = result.trades.executed_trade_amount.abs().idxmax()
    row = result.trades.loc[traded]
    quote = market.loc[market.trade_date.eq(row.execution_date) & market.instrument.eq(row.instrument)].iloc[0]
    inflated = float(row.executed_trade_amount) * 1.5
    result.trades.loc[traded, "executed_trade_amount"] = inflated
    for field, value in trade_cost(inflated, quote.volatility, quote.daily_amount_cny,
                                   row.execution_date, model).items():
        result.trades.loc[traded, field] = value
    with pytest.raises(DiagnosticError, match="outside the tradable leg range"):
        reconstruct_executed_path(daily, result.trades, market, 10_000, model)


def test_corrupted_blocked_flag_fails_closed():
    market, _, model, result, daily, _ = synthetic_case()
    result.trades.loc[0, "blocked_by_tradability"] = not result.trades.loc[0, "blocked_by_tradability"]
    with pytest.raises(DiagnosticError, match="blocked_by_tradability"):
        reconstruct_executed_path(daily, result.trades, market, 10_000, model)


def test_mark_fallback_uses_last_close_never_future_or_last_vwap():
    dates = pd.bdate_range("2023-01-02", periods=4)
    market = pd.DataFrame({"trade_date": dates, "instrument": "A", "close": [10., np.nan, np.nan, 999.],
                           "vwap": [9., 12., np.nan, 888.]})
    mark, execution = price_panels(market)
    assert mark.A.tolist() == [10, 12, 10, 999]
    assert execution.A.tolist() == [9, 12, 10, 888]
    market.loc[3, ["close", "vwap"]] = [1., 2.]
    changed, _ = price_panels(market)
    pd.testing.assert_frame_equal(mark.iloc[:3], changed.iloc[:3])


def test_float32_qlib_prices_do_not_leak_into_reconstructed_returns():
    """Qlib serves float32 quotes; mark/execution ratios must be widened first.

    In float32 the near-1.0 quotient loses ~1e-5 of the intraday return, which
    prices out as a ~1e-9 fraction of NAV against a full rebalance notional.
    NAV stays exact either way, so only the derived identity would catch it.
    """
    market, targets, model, _, _, _ = synthetic_case()
    market[["close", "vwap"]] = market[["close", "vwap"]].astype("float32")
    result = run_vwap_backtest(market, targets, 10_000., model)
    dates = sorted(market.trade_date.unique())
    _, daily = summarize_backtest_performance(result.nav, pd.DataFrame({"trade_date": dates, "close": 100.}), initial_capital=10_000.)
    mark, execution = price_panels(market)
    assert mark.to_numpy().dtype == np.float64 and execution.to_numpy().dtype == np.float64
    path = reconstruct_executed_path(daily, result.trades, market, 10_000., model)
    assert path.daily.accounting_residual.abs().max() < 1e-12
    widened = market.copy()
    widened[["close", "vwap"]] = widened[["close", "vwap"]].astype("float64")
    pd.testing.assert_frame_equal(price_panels(widened)[0], mark)
    pd.testing.assert_frame_equal(price_panels(widened)[1], execution)


def test_float32_qlib_prices_do_not_leak_into_attribution():
    market, targets, model, _, _, exposures = synthetic_case()
    market[["close", "vwap"]] = market[["close", "vwap"]].astype("float32")
    result = run_vwap_backtest(market, targets, 10_000., model)
    dates = sorted(market.trade_date.unique())
    _, daily = summarize_backtest_performance(result.nav, pd.DataFrame({"trade_date": dates, "close": 100.}), initial_capital=10_000.)
    path = reconstruct_executed_path(daily, result.trades, market, 10_000., model)
    attributed, _, _, _, _ = attribute_executed_path(daily, path, market, exposures,
        ["industry_A", "industry_B", "industry_absent"], ["style_Size", "style_Beta"])
    assert attributed.accounting_residual.abs().max() < 1e-12


def test_suspension_with_missing_quotes_reconciles_causal_marks():
    market, targets, model, _, _, _ = synthetic_case()
    dates = sorted(market.trade_date.unique())
    selector = market.instrument.eq("S00") & market.trade_date.isin(dates[2:6])
    market.loc[selector, ["vwap", "close"]] = np.nan
    market.loc[selector, ["can_buy", "can_sell"]] = False
    result = run_vwap_backtest(market, targets, 10_000., model)
    _, daily = summarize_backtest_performance(result.nav, pd.DataFrame({"trade_date": dates, "close": 100.}), initial_capital=10_000.)
    path = reconstruct_executed_path(daily, result.trades, market, 10_000, model)
    assert path.daily.accounting_residual.abs().max() < 1e-12


def test_wls_uses_sqrt_cap_objective_and_handles_absent_industry():
    _, _, _, _, _, exposure = synthetic_case()
    exposure = exposure.loc[exposure.trade_date.eq(exposure.trade_date.min())].set_index("instrument")
    industries = ["industry_A", "industry_B", "industry_absent"]
    styles = ["style_Size", "style_Beta"]
    x = exposure[industries[:2] + styles].to_numpy()
    y = pd.Series(np.arange(len(x)) ** 2 / 1000, index=exposure.index)
    f, audit = wls_cross_section(exposure, y, industries, styles)
    w = np.diag(np.sqrt(exposure.market_cap))
    expected = np.linalg.solve(x.T @ w @ x, x.T @ w @ y)
    np.testing.assert_allclose(f[industries[:2] + styles], expected, rtol=1e-10)
    assert f.industry_absent == 0 and audit["rank"] == 4
    wrong = np.linalg.solve(x.T @ np.diag(exposure.market_cap) @ x, x.T @ np.diag(exposure.market_cap) @ y)
    assert np.max(np.abs(expected - wrong)) > 1e-6


def test_rank_deficiency_is_not_hidden_with_ridge():
    _, _, _, _, _, exposure = synthetic_case()
    exposure = exposure.loc[exposure.trade_date.eq(exposure.trade_date.min())].set_index("instrument")
    exposure.style_Beta = exposure.style_Size
    with pytest.raises(DiagnosticError, match="rank-deficient"):
        wls_cross_section(exposure, pd.Series(.01, index=exposure.index), ["industry_A", "industry_B"], ["style_Size", "style_Beta"])


def test_exact_monthly_link_is_not_sum_or_compound_daily_excess():
    frame = pd.DataFrame({"trade_date": pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01"]),
        "portfolio_return": [.2, -.1, .03], "benchmark_return": [.1, .05, .02],
        "industry": [.03, -.07, .004], "specific": [.07, -.08, .006]})
    result = linked_monthly(frame, ["industry", "specific"])
    assert result.excess_return.iloc[0] == pytest.approx(1.2 * .9 - 1.1 * 1.05)
    np.testing.assert_allclose(result.industry + result.specific, result.excess_return, atol=1e-14)
    assert result.excess_return.iloc[0] != pytest.approx((frame.portfolio_return - frame.benchmark_return).iloc[:2].sum())


def test_attribution_reconciles_and_keeps_official_index_bridge_out_of_specific():
    market, _, model, result, daily, exposures = synthetic_case()
    path = reconstruct_executed_path(daily, result.trades, market, 10_000, model)
    args = (daily, path, market, exposures, ["industry_A", "industry_B", "industry_absent"], ["style_Size", "style_Beta"])
    d, monthly, _, audit, summary = attribute_executed_path(*args)
    assert (audit.exposure_date < audit.trade_date).all()
    assert set(audit.period) == {"close_to_close", "execution_to_close"}
    assert d.accounting_residual.abs().max() < 1e-12
    assert summary["decision"] == "No-Go"
    changed = daily.copy(); changed.benchmark_return += .001
    changed_d, _, _, _, _ = attribute_executed_path(changed, *args[1:])
    np.testing.assert_allclose(d.specific_gross, changed_d.specific_gross, atol=1e-14)
    np.testing.assert_allclose(changed_d.benchmark_bridge - d.benchmark_bridge, -.001, atol=1e-14)
    np.testing.assert_allclose(d.specific_net, d.specific_gross + d.cost, atol=1e-14)
    assert monthly.three_block_residual.abs().sum() > 0


def test_future_exposure_perturbation_cannot_change_earlier_attribution():
    market, _, model, result, daily, exposures = synthetic_case()
    path = reconstruct_executed_path(daily, result.trades, market, 10_000, model)
    args = (daily, path, market)
    columns = (["industry_A", "industry_B"], ["style_Size", "style_Beta"])
    first = attribute_executed_path(*args, exposures, *columns)[0]
    cutoff = daily.trade_date.iloc[5]
    changed = exposures.copy()
    changed.loc[changed.trade_date.ge(cutoff), "style_Size"] *= 100
    second = attribute_executed_path(*args, changed, *columns)[0]
    pd.testing.assert_frame_equal(first.loc[first.trade_date.le(cutoff)], second.loc[second.trade_date.le(cutoff)])


def test_weekly_only_exposures_rejected_for_daily_attribution():
    market, _, model, result, daily, exposures = synthetic_case()
    path = reconstruct_executed_path(daily, result.trades, market, 10_000, model)
    with pytest.raises(DiagnosticError, match="DAILY"):
        attribute_executed_path(daily, path, market, exposures.loc[exposures.trade_date.dt.dayofweek.eq(4)],
                               ["industry_A", "industry_B"], ["style_Size", "style_Beta"])


def test_turnover_definitions_respect_forced_exits_and_cash_legs():
    split = turnover_split(pd.Series({"A": .6, "B": .4}), pd.Series({"A": .7, "OLD": .3}), ["A", "B"])
    assert split == pytest.approx({"target_total_turnover": .4, "target_mandatory_turnover": .3, "target_discretionary_turnover": .1})
    orders = pd.DataFrame({"instrument": ["OLD", "A"], "executed_trade_amount": [-30., 10.], "desired_trade_amount": [-30., 30.]})
    actual = executed_turnover_split(orders, ["A"], 100.)
    assert actual["executed_total_turnover"] == .2
    assert actual["executed_mandatory_sell_leg"] == .15
    assert actual["executed_other_legs"] == pytest.approx(.05)
    assert actual["execution_shortfall_turnover"] == .1
    assert half_l1(pd.Series({"A": .8, "CASH": .2}), pd.Series({"A": 1.})) == pytest.approx(.2)


def test_gross_net_on_realized_path_preserves_both_annualization_conventions():
    _, _, _, _, daily, _ = synthetic_case()
    metrics = path_gross_net(daily, 10_000.)
    assert metrics["gross_excess_cagr_difference"] - metrics["cost_drag_cagr_difference"] == pytest.approx(metrics["net_excess_cagr_difference"])
    assert metrics["gross_excess_monthly_annualized"] - metrics["cost_drag_monthly_annualized"] == pytest.approx(metrics["net_excess_monthly_annualized"])
    assert metrics["transaction_cost_cny"] > 0


def test_delay_uses_trading_days_and_records_cutoff_without_new_data():
    calendar = pd.to_datetime(["2026-02-13", "2026-02-24", "2026-02-25", "2026-02-26"])
    target = pd.DataFrame({"signal_date": [calendar[0], calendar[-2]], "instrument": "A", "target_weight": 1.})
    dispatched, schedule = stress_schedule(target, calendar, "delay_one_day")
    assert dispatched.execution_date.tolist() == [calendar[2]]
    assert dispatched.signal_date.tolist() == [calendar[1]]
    assert schedule.status.tolist() == ["scheduled", "beyond_cutoff"]
    assert schedule.source_signal_date.tolist() == target.signal_date.tolist()


def test_biweekly_parity_does_not_reset_for_missing_week_or_new_year():
    calendar = pd.bdate_range("2025-12-19", "2026-01-30")
    signals = pd.to_datetime(["2025-12-19", "2026-01-02", "2026-01-09", "2026-01-16"])
    targets = pd.DataFrame({"signal_date": signals, "instrument": "A", "target_weight": 1.})
    dispatched, _ = stress_schedule(targets, calendar, "biweekly")
    assert dispatched.source_signal_date.tolist() == list(pd.to_datetime(["2025-12-19", "2026-01-02", "2026-01-16"]))


def test_delayed_initial_cash_day_is_kept_against_main_benchmark():
    market, targets, model, _, main, _ = synthetic_case()
    dispatched, _ = stress_schedule(targets, sorted(market.trade_date.unique()), "delay_one_day")
    result = run_vwap_backtest(market, dispatched[["signal_date", "instrument", "target_weight"]], 10_000., model)
    stress = align_stress_daily(result.nav, main, 10_000.)
    assert stress.trade_date.equals(main.trade_date)
    assert stress.portfolio_return.iloc[0] == 0
    assert stress.cash.iloc[0] == 10_000
    assert stress.benchmark_return.equals(main.benchmark_return)
    reconstruct_executed_path(stress, result.trades, market, 10_000., model)


def test_inherited_label_availability_changes_prediction_sample_membership():
    """Register a real upstream limitation WITHOUT fixing frozen D4-D9."""
    calendar = pd.bdate_range("2018-01-02", periods=10)
    prices = pd.DataFrame([{"trade_date": d, "instrument": n, "vwap": 10. + i}
                          for i, d in enumerate(calendar) for n in ["A", "B"]])
    weights = pd.DataFrame({"trade_date": calendar[0], "instrument": ["A", "B"], "weight": [.5, .5]})
    original, _ = build_forward_excess_labels(prices, weights, [calendar[0]], calendar, [5])
    future_missing = prices.copy()
    future_missing.loc[future_missing.trade_date.eq(calendar[6]) & future_missing.instrument.eq("A"), "vwap"] = np.nan
    changed, _ = build_forward_excess_labels(future_missing, weights, [calendar[0]], calendar, [5])
    features = weights.rename(columns={"trade_date": "sample_date"}).assign(feature=1.)
    # The same inner join is used before prediction in D5 and D9.
    assert set(features.merge(original, on=["sample_date", "instrument"]).instrument) == {"A", "B"}
    assert set(features.merge(changed, on=["sample_date", "instrument"]).instrument) == {"B"}
    pd.testing.assert_frame_equal(prices.loc[prices.trade_date.le(calendar[0])], future_missing.loc[future_missing.trade_date.le(calendar[0])])


def test_output_namespace_cannot_overwrite_escape_or_follow_symlink(tmp_path):
    output = D10Output(tmp_path, "test_v1")
    output.json("summary.json", {"decision": "No-Go"})
    with pytest.raises(FileExistsError):
        output.json("summary.json", {})
    with pytest.raises(FileExistsError):
        D10Output(tmp_path, "test_v1")
    with pytest.raises(DiagnosticError):
        D10Output(tmp_path, "../d9_v2")
    with pytest.raises(DiagnosticError):
        output.text("../../d9_v2_summary.json", "bad")
    output.path.joinpath("link.json").symlink_to(tmp_path / "d9.json")
    with pytest.raises(FileExistsError):
        output.json("link.json", {})


def test_hash_guard_detects_edit_add_delete_including_raw_provider(tmp_path):
    (tmp_path / "reports").mkdir(); (tmp_path / "reports/d9_v2_summary.json").write_text("original")
    provider = tmp_path / "provider"; (provider / "features/a").mkdir(parents=True)
    (provider / "features/a/close.day.bin").write_bytes(b"old-price")
    config = {"data": {"qlib_provider_uri": str(provider)}}
    before = snapshot_inputs(tmp_path, config)
    D10Output(tmp_path, "v1").text("report.md", "diagnostic")
    assert_snapshot_unchanged(before, snapshot_inputs(tmp_path, config))
    (provider / "features/a/close.day.bin").write_bytes(b"new-price")
    with pytest.raises(DiagnosticError, match="features/a/close.day.bin"):
        assert_snapshot_unchanged(before, snapshot_inputs(tmp_path, config))
    with pytest.raises(DiagnosticError):
        assert_snapshot_unchanged(before, {})


def test_repository_guard_rejects_branch_and_historical_mutation(tmp_path):
    def command(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    command("init", "-b", "d10")
    command("config", "user.name", "D10 fixture"); command("config", "user.email", "test@example.invalid")
    (tmp_path / "src").mkdir(); old = tmp_path / "src/d9_core.py"; old.write_text("# frozen\n")
    command("add", "."); command("commit", "-m", "synthetic baseline")
    command("tag", "-a", "frozen-oos", "-m", "synthetic freeze")
    baseline = {"base_commit": command("rev-parse", "HEAD"), "frozen_commit": command("rev-parse", "HEAD"),
        "frozen_tag_object": command("rev-parse", "frozen-oos"), "protected_blobs": {"src/d9_core.py": git_blob(old)}}
    assert verify_repository(tmp_path, baseline)["branch"] == "d10"
    command("switch", "-c", "other")
    with pytest.raises(DiagnosticError, match="d10 branch"):
        verify_repository(tmp_path, baseline)
    command("switch", "d10"); old.write_text("# modified\n")
    with pytest.raises(DiagnosticError, match="D1-D9 tracked"):
        verify_repository(tmp_path, baseline)


def test_unsafe_nan_does_not_pass_numeric_gate():
    with pytest.raises(DiagnosticError):
        require_close([np.nan], [np.nan], "nan gate")

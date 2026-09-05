"""Small complete orchestration fixture; only local Qlib ingestion is replaced.

Uses frozen D3 matching, actual Parquet I/O, both capitals/seven layers,
attribution, reused impact2x and new schedule stresses. No production data.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import d10_pipeline as pipeline
from scripts.d7_pipeline import build_execution_price_matrix
from scripts.d8_pipeline import build_frozen_d7_pretrade_path
from src.d3_core import CostModel, run_vwap_backtest
from src.d8_core import make_ideal_execution_market, summarize_backtest_performance
from src.d9_core import summarize_monthly_performance
from src.d10_core import half_l1
from src.d10_io import D10Output, sha256
from src.data_access import ResearchDataAccess
from src.d10_core import DiagnosticError
from src.d6_core import compute_raw_style_history
from scripts.d6_pipeline import assemble_exposures


@pytest.mark.parametrize("hostile", [False, True], ids=["clean", "hostile"])
def test_entire_diagnostic_orchestration_preserves_inputs_and_nogo(tmp_path, monkeypatch, hostile):
    """The hostile pass carries every awkward shape the real ladder contains.

    Each of these cost a live run before it was covered here: Qlib float32
    quotes, a name with no bar at all (a long suspension), a suspended bar with
    no close or vwap, a side blocked by the D2 amount floor, a binding
    participation cap, and a cheap stock held in size whose exit leaves negative
    unit dust.
    """
    config = yaml.safe_load((pipeline.ROOT / "config/config.yaml").read_text())
    config["windows"].update(design_start="2017-01-01", design_end="2017-08-31", oos_start="2025-12-01", oos_end="2026-07-31")
    (tmp_path / "config").mkdir(); (tmp_path / "config/config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "data").mkdir(); (tmp_path / config["data"]["window_marker"]).write_text('{"mode":"oos"}')
    reports = tmp_path / "reports"; reports.mkdir()
    base = CostModel.from_config(config)
    windows = {}
    for window in ["design", "oos"]:
        rng = np.random.default_rng(901 if window == "design" else 902)
        start, end = [pd.Timestamp(config["windows"][f"{window}_{x}"]) for x in ["start", "end"]]
        dates = pd.DatetimeIndex([d for m in pd.period_range(start, end, freq="M")
                                 for d in pd.bdate_range(m.start_time, periods=3)])
        n = 12; names = [f"S{i:02}" for i in range(n)]
        closes = 10 * np.cumprod(1 + rng.normal(.001, .02, (len(dates), n)), axis=0)
        vwaps = closes / (1 + rng.normal(0, .004, closes.shape))
        market = pd.DataFrame([{"trade_date": d, "instrument": name, "close": closes[t, i],
            "vwap": vwaps[t, i], "daily_amount_cny": 100_000_000., "volatility": .02,
            "can_buy": True, "can_sell": not (t == 7 and i < 3)}
            for t, d in enumerate(dates) for i, name in enumerate(names)])
        if hostile:
            pick = lambda name, days: market.instrument.eq(name) & market.trade_date.isin(days)
            market.loc[pick(names[1], dates[10:12]), ["close", "vwap"]] = np.nan
            market.loc[pick(names[2], dates[7:20]), "can_buy"] = False
            market.loc[pick(names[4], dates), "daily_amount_cny"] = 60_000.
            market.loc[pick(names[5], dates), ["close", "vwap"]] /= 8
            market[["close", "vwap"]] = market[["close", "vwap"]].astype("float32")
            market = market.loc[~pick(names[0], dates[4:7])].reset_index(drop=True)
        ideal = make_ideal_execution_market(market, fallback_daily_amount=50_000_000.)
        weights = pd.DataFrame([{"trade_date": d, "instrument": name, "weight": 1 / n} for d in dates for name in names])
        exposure = weights.rename(columns={"weight": "benchmark_weight"}).copy()
        x = rng.normal(size=(n, 6))
        styles = ["style_" + s for s in config["d6"]["style_factors"]]
        for i, col in enumerate(styles): exposure[col] = np.tile(x[:, i], len(dates))
        exposure["industry_A"] = np.tile(np.arange(n) < n // 2, len(dates)).astype(float)
        exposure["industry_B"] = 1 - exposure.industry_A
        exposure["market_cap"] = np.tile(100 + np.arange(n) ** 2, len(dates))
        target = pd.DataFrame([{"signal_date": d, "execution_date": dates[t + 1], "instrument": name,
            "target_weight": (2 / n) * (i % 2 == (t // 3) % 2)}
            for t, d in enumerate(dates) if t % 3 == 0 for i, name in enumerate(names)])
        targets = {i: target.copy() for i in range(7)}
        schedule = target[["signal_date", "execution_date"]].drop_duplicates().set_index("signal_date").execution_date
        refs = build_frozen_d7_pretrade_path(target, exposure, schedule.index, schedule,
                                            build_execution_price_matrix(market), tolerance=1e-6)
        audits = pd.DataFrame([{"layer": layer, "signal_date": s,
            "target_single_sided_turnover": half_l1(g.set_index("instrument").target_weight, refs[s])}
            for layer in range(7) for s, g in target.groupby("signal_date")])
        prefix = "d8" if window == "design" else "d9_v2"
        benchmark = pd.DataFrame({"trade_date": dates, "close": closes.mean(axis=1)})
        benchmark["benchmark_return"] = benchmark.close.pct_change(fill_method=None)
        expected = []
        for layer in range(7):
            daily_parts, trade_parts = [], []
            model = CostModel.zero_cost(1.) if layer == 0 else replace(base, participation_limit=1.) if layer == 1 else base
            for capital in [10_000_000., 20_000_000.]:
                result = run_vwap_backtest(ideal if layer < 2 else market, target, capital, model)
                metrics, daily = summarize_backtest_performance(result.nav, benchmark, initial_capital=capital)
                daily["capital"] = capital; daily["layer"] = layer
                result.trades["capital"] = capital; result.trades["layer"] = layer
                daily_parts.append(daily); trade_parts.append(result.trades)
                if capital == 10_000_000.:
                    expected.append({"layer": layer, "annualized_excess_return": metrics["annualized_excess_return"]})
                    if layer == 6 and window == "oos":
                        monthly_metrics, monthly = summarize_monthly_performance(daily, newey_west_lags=[3, 4, 5, 6])
                        daily.to_parquet(reports / "d9_v2_main_daily.parquet", index=False)
                        monthly.to_parquet(reports / "d9_v2_main_monthly.parquet", index=False)
                        stress = run_vwap_backtest(market, target, capital, replace(base, impact_coefficient=base.impact_coefficient * 2))
                        _, stress_daily = summarize_backtest_performance(stress.nav, benchmark, initial_capital=capital)
                        stress_daily.to_parquet(reports / "d9_v2_impact2x_daily.parquet", index=False)
                        stress.trades.to_parquet(reports / "d9_v2_impact2x_trades.parquet", index=False)
            pd.concat(daily_parts).to_parquet(reports / f"{prefix}_layer_{layer}_daily.parquet", index=False)
            pd.concat(trade_parts).to_parquet(reports / f"{prefix}_layer_{layer}_trades.parquet", index=False)
        payload = {"layers": expected} if window == "design" else {"oos_ladder": expected, "performance": monthly_metrics}
        (reports / f"{prefix}_summary.json").write_text(json.dumps(payload))
        windows[window] = dict(start=start, end=end, prefix=prefix, weights=weights, flags=market,
            market=market, strategy=market, ideal=ideal, exposures=exposure, industries=["industry_A", "industry_B"],
            styles=styles, targets=targets, references={0: refs, 6: refs}, executions=schedule,
            target_audit=audits, calendar=dates, official=benchmark)
    hashes = {str(p): sha256(p) for p in reports.iterdir()}
    monkeypatch.setattr(pipeline, "load_window", lambda access, cfg, window, output: windows[window])
    output = D10Output(tmp_path, "integration_v1")
    summary, ladder = pipeline.run_diagnostics(ResearchDataAccess(tmp_path), config, output)
    assert len(ladder) == 28
    assert len(summary["ladder2_comparisons"]) == 4
    assert set(summary["attribution"]) == {"design", "oos"}
    assert len(summary["stress_scenarios"]) == 3
    assert summary["decision"] == "No-Go" and not summary["decision_changed"]
    july = summary["july_2026"]
    assert july["month"] == "2026-07"
    np.testing.assert_allclose(july["portfolio_return"] - july["benchmark_return"], july["excess_return"])
    assert hashes == {p: sha256(p) for p in hashes}
    summary["diagnostic_acceptance"]["passed"] = summary["diagnostic_acceptance"].pop("passed_before_final_input_integrity_check")
    pipeline.write_report(summary, ladder, output)
    output.json("d10_summary.json", summary)
    assert (output.path / "d10_report.md").is_file()
    assert (output.path / "oos_attribution_monthly.png").stat().st_size > 1000


def test_read_years_clips_last_year_at_oos_cutoff(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    pd.DataFrame({"trade_date": pd.to_datetime(["2026-08-31", "2026-09-01"]), "value": [1, 999]}).to_parquet(raw / "daily_basic_2026.parquet")
    class Access:
        root = tmp_path
        requests = []
        def read_parquet(self, relative, *, date_col, start_date, end_date):
            self.requests.append(pd.Timestamp(end_date))
            frame = pd.read_parquet(self.root / relative)
            return frame.loc[frame[date_col].between(start_date, end_date)]
    access = Access()
    result = pipeline.read_years(access, {"data": {"tushare_raw_dir": "raw"}}, "daily_basic", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-31"))
    assert result.value.tolist() == [1]
    assert access.requests == [pd.Timestamp("2026-08-31")]


def test_real_window_loader_rebuilds_daily_exposures_and_rejects_weekly_drift(tmp_path):
    """Exercise actual file schema/ingestion; replace only Qlib feature transport."""
    config = yaml.safe_load((pipeline.ROOT / "config/config.yaml").read_text())
    config["windows"].update(oos_start="2019-01-01", oos_end="2019-02-28")
    (tmp_path / "config").mkdir(); (tmp_path / "config/config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "data").mkdir(); (tmp_path / config["data"]["window_marker"]).write_text('{"mode":"oos"}')
    def save(relative, frame):
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-28")
    history = start - pd.Timedelta(days=config["d6"]["history_buffer_calendar_days"])
    dates = pd.bdate_range(history, end); n = 40
    rng = np.random.default_rng(171)
    names = [f"SH600{i:03}" for i in range(n)]
    close = 10 * np.cumprod(1 + rng.normal(.0001, .01, (len(dates), n)), axis=0)
    market = pd.DataFrame({"trade_date": np.repeat(dates, n), "instrument": np.tile(names, len(dates)), "close": close.ravel()})
    market["vwap"] = market.close * .999
    for c in ["open", "high", "low"]: market[c] = market.close
    market["factor"] = 1.; market["volume"] = 1_000_000.; market["amount"] = 100_000.
    basic = market[["trade_date", "instrument"]].copy()
    basic["total_mv"] = np.exp(rng.normal(12, .5, len(basic)))
    basic["turnover_rate"] = rng.uniform(.1, 4, len(basic))
    benchmark = pd.DataFrame({"trade_date": dates, "close": 100 * np.cumprod(1 + rng.normal(0, .008, len(dates)))})
    benchmark["benchmark_return"] = benchmark.close.pct_change(fill_method=None)
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    save(raw_dir / "index_daily_H00905_CSI.parquet", benchmark)
    for year, frame in basic.groupby(basic.trade_date.dt.year): save(raw_dir / f"daily_basic_{year}.parquet", frame)
    current = market.loc[market.trade_date.ge(start)]
    limits = current[["trade_date", "instrument"]].copy()
    limits["ts_code"] = limits.instrument.str[2:] + ".SH"
    limits["up_limit"] = 1e6; limits["down_limit"] = .001
    save(raw_dir / "stk_limit_2019.parquet", limits)
    save(raw_dir / "suspend_d_2019.parquet", pd.DataFrame({"trade_date": pd.Series(dtype="datetime64[ns]"), "ts_code": pd.Series(dtype=str), "suspend_type": pd.Series(dtype=str)}))
    weights = current[["trade_date", "instrument"]].assign(weight=1/n)
    flags = current[["trade_date", "instrument"]].assign(can_buy=True, can_sell=True)
    processed = Path(config["d9"]["v2"]["processed_oos_dir"])
    save(processed / "benchmark_weights.parquet", weights); save(processed / "tradability_flags.parquet", flags)
    catalog = pd.DataFrame({"l1_code": [f"801{i:03}.SI" for i in range(31)], "industry_name": [f"I{i}" for i in range(31)]})
    catalog["industry_factor"] = "industry_" + catalog.l1_code.str.replace(".", "_", regex=False)
    mapping = pd.DataFrame({"instrument": names, "l1_code": np.tile(catalog.l1_code.iloc[:2], n//2),
        "industry_name": np.tile(catalog.industry_name.iloc[:2], n//2), "industry_factor": np.tile(catalog.industry_factor.iloc[:2], n//2)})
    save(config["d6"]["industry_mapping_file"], mapping); save(config["d6"]["industry_catalog_file"], catalog)
    d6 = config["d6"]
    raw_styles = compute_raw_style_history(market[["trade_date", "instrument", "close"]], benchmark, basic,
        beta_window=d6["beta_window_trade_days"], beta_minimum_observations=d6["beta_minimum_observations"],
        momentum_window=d6["momentum_window_trade_days"], momentum_skip=d6["momentum_skip_trade_days"],
        liquidity_window=d6["liquidity_window_trade_days"], liquidity_minimum_observations=d6["liquidity_minimum_observations"])
    exposure, _, _, _ = assemble_exposures(weights, raw_styles, mapping, catalog,
        pd.DatetimeIndex(weights.trade_date.unique()), winsor_mad_multiple=d6["winsor_mad_multiple"])
    signals = pd.to_datetime(["2019-01-04", "2019-02-01"])
    weekly = exposure.loc[exposure.trade_date.isin(signals)]
    save("reports/d9_v2_factor_exposures.parquet", weekly)
    targets = pd.DataFrame([{"signal_date": s, "execution_date": s + pd.offsets.BDay(), "instrument": name, "target_weight": 1/n} for s in signals for name in names])
    save("reports/d9_v2_target_weights.parquet", targets)
    for layer in range(7): save(f"reports/d9_v2_layer_{layer}_targets.parquet", targets)
    save("reports/d9_v2_ladder_target_audit.parquet", pd.DataFrame({"signal_date": signals, "layer": 6}))
    save("reports/d9_v2_oos_predictions.parquet", targets.rename(columns={"signal_date": "sample_date"}))
    class Access(ResearchDataAccess):
        def features(self, instruments, fields, start_date, end_date, freq="day"):
            self.assert_window(start_date, end_date)
            assert pd.Timestamp(end_date) <= end
            frame = market.loc[market.trade_date.between(start_date, end_date) & market.instrument.isin(instruments)].rename(columns={"trade_date": "datetime"}).set_index(["instrument", "datetime"])
            result = frame[[f.removeprefix("$") for f in fields]].copy(); result.columns = fields
            return result
    access = Access(tmp_path)
    output = D10Output(tmp_path, "loader_v1")
    loaded = pipeline.load_window(access, config, "oos", output)
    assert loaded["exposures"].trade_date.nunique() > weekly.trade_date.nunique()
    assert len(loaded["industries"]) == 31 and len(loaded["styles"]) == 6
    parity = json.loads((output.path / "oos_exposure_parity.json").read_text())
    assert parity["passed"]
    assert (output.path / "oos_prediction_universe_audit.parquet").is_file()
    weekly = weekly.copy(); weekly.loc[weekly.index[0], "style_Size"] += .01
    save("reports/d9_v2_factor_exposures.parquet", weekly)
    with pytest.raises(DiagnosticError, match="weekly exposure parity"):
        pipeline.load_window(access, config, "oos", D10Output(tmp_path, "loader_bad"))

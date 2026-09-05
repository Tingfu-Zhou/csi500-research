"""D10 read-only probe: locate why a frozen order disagrees with the rebuilt panel.

Prints evidence to stdout only. Creates no run directory, writes no file, touches
no D1-D9 artefact. Safe to run repeatedly; it is not part of the D10 acceptance
path and produces no research number.

    python -u scripts/d10_probe_tradability.py [--window design] [--date 2017-02-13]
        [--instrument SZ002073]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402
from scripts.d3_pipeline import (  # noqa: E402
    add_ex_ante_volatility, add_exchange_flags, load_qlib_market, load_yearly, prepare_markets,
)

pd.set_option("display.width", 200, "display.max_columns", 60)


def read_parquet(access, path, date_col, start, end):
    frame = access.read_parquet(path, date_col=date_col, start_date=start, end_date=end)
    frame[date_col] = parse_date_series(frame[date_col])
    return frame


def window_trimmed_years(access, config, endpoint, start, end):
    """What D10's read_years does: every year file trimmed to the WINDOW."""
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = Path(config["data"]["tushare_raw_dir"]) / f"{endpoint}_{year}.parquet"
        frames.append(read_parquet(access, path, "trade_date", start, end))
    return pd.concat(frames, ignore_index=True)


def year_trimmed_years(access, config, endpoint, start, end):
    """What D8's load_yearly does: every year file trimmed to ITS OWN year."""
    return load_yearly(access, config, endpoint, pd.Timestamp(start).year, pd.Timestamp(end).year)


def report_year_leakage(access, config, endpoint, start, end):
    print(f"\n### {endpoint}: rows carried outside their own year file")
    leaked = 0
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = Path(config["data"]["tushare_raw_dir"]) / f"{endpoint}_{year}.parquet"
        frame = read_parquet(access, path, "trade_date", start, end)
        outside = frame.loc[frame.trade_date.dt.year.ne(year)]
        if len(outside):
            leaked += len(outside)
            span = f"{outside.trade_date.min().date()}..{outside.trade_date.max().date()}"
            print(f"  {endpoint}_{year}.parquet: {len(outside)} rows dated {span}")
    print(f"  total leaked rows: {leaked}"
          + ("  <-- D10 read_years would double-count these; D8 load_yearly would not" if leaked else "  (none)"))
    return leaked


def load_raw_market(access, config, window):
    """load_window's own ingestion, up to but not including the flag joins."""
    start = pd.Timestamp(config["windows"][f"{window}_start"])
    end = pd.Timestamp(config["windows"][f"{window}_end"])
    processed = Path(config["data"]["processed_design_dir"] if window == "design"
                     else config["d9"]["v2"]["processed_oos_dir"])
    weights = read_parquet(access, processed / "benchmark_weights.parquet", "trade_date", start, end)
    flags = read_parquet(access, processed / "tradability_flags.parquet", "trade_date", start, end)
    instruments = sorted(weights.instrument.astype(str).unique())
    history_start = start if window == "design" else start - pd.Timedelta(
        days=int(config["d6"]["history_buffer_calendar_days"]))
    market = load_qlib_market(access, instruments, history_start, end,
                              float(config["tradability"]["qlib_amount_multiplier_to_cny"]))
    return market, flags, start, end


def build_panel(config, raw_market, flags, start, end, limits, suspensions):
    market = add_exchange_flags(raw_market.copy(), limits, suspensions, config)
    market = add_ex_ante_volatility(market, config)
    strategy, _ = prepare_markets(market, flags)
    strategy = strategy.loc[strategy.trade_date.between(start, end)].copy()
    return market, strategy.set_index(["trade_date", "instrument"]).sort_index()


def section(label, function, *args, **kwargs):
    """Never let one failing probe section hide the others."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not decide
        import traceback
        print(f"\n!! probe section {label!r} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="read-only D10 tradability probe")
    parser.add_argument("--window", default="design", choices=["design", "oos"])
    parser.add_argument("--date", default="2017-02-13")
    parser.add_argument("--instrument", default="SZ002073")
    parser.add_argument("--max-rows", type=int, default=25)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    access = ResearchDataAccess(ROOT)
    window = args.window
    start = pd.Timestamp(config["windows"][f"{window}_start"])
    end = pd.Timestamp(config["windows"][f"{window}_end"])
    key = (pd.Timestamp(args.date), args.instrument)
    print(f"# D10 tradability probe: window={window} [{start.date()}..{end.date()}] "
          f"target={args.instrument} @ {pd.Timestamp(args.date).date()}")

    leaked = sum(section(f"leakage:{endpoint}", report_year_leakage, access, config, endpoint, start, end) or 0
                 for endpoint in ["stk_limit", "suspend_d"])

    print("\n## rebuilding the strategy panel two ways")
    d10_limits = window_trimmed_years(access, config, "stk_limit", start, end)
    d10_susp = window_trimmed_years(access, config, "suspend_d", start, end)
    d8_limits = year_trimmed_years(access, config, "stk_limit", start, end)
    d8_susp = year_trimmed_years(access, config, "suspend_d", start, end)
    print(f"  stk_limit rows: D10 read_years={len(d10_limits)}  D8 load_yearly={len(d8_limits)}")
    print(f"  suspend_d rows: D10 read_years={len(d10_susp)}  D8 load_yearly={len(d8_susp)}")

    raw_market, flags, start, end = load_raw_market(access, config, window)
    market, panel_d10 = build_panel(config, raw_market, flags, start, end, d10_limits, d10_susp)
    _, panel_d8 = build_panel(config, raw_market, flags, start, end, d8_limits, d8_susp)
    if not panel_d10.index.equals(panel_d8.index):
        raise SystemExit("the two panels do not even share an index; report this line")
    differ = (panel_d10.can_buy.ne(panel_d8.can_buy) | panel_d10.can_sell.ne(panel_d8.can_sell))
    print(f"  panel rows: {len(panel_d10)};  can_buy/can_sell differ on {int(differ.sum())} rows")
    if differ.any():
        print(panel_d10.loc[differ, ["can_buy", "can_sell"]].join(
            panel_d8.loc[differ, ["can_buy", "can_sell"]], rsuffix="_d8_style").head(args.max_rows))

    section("target row dump", dump_target_rows, key, args, market, flags, panel_d10, panel_d8)
    section("raw endpoint rows", dump_raw_endpoint_rows, args, d10_limits, d8_limits, d10_susp, d8_susp)
    total = section("frozen order scan", scan_orders, access, config, window, start, end, panel_d10, args) or 0
    print(f"\n# summary: leaked year rows={leaked}; panel rows differing={int(differ.sum())}; "
          f"rejected frozen orders in layers 2-6={total}")


def dump_target_rows(key, args, market, flags, panel_d10, panel_d8):
    print(f"\n## every frozen input behind {key}")
    market_key = market.set_index(["trade_date", "instrument"]).sort_index()
    for label, frame, cols in [
        ("qlib market + exchange flags", market_key,
         ["close", "vwap", "raw_high", "raw_low", "raw_close", "up_limit", "down_limit",
          "daily_amount_cny", "volatility", "is_suspended", "is_one_price_limit_up",
          "is_one_price_limit_down", "exchange_can_buy", "exchange_can_sell"]),
        ("D2 tradability flags", flags.set_index(["trade_date", "instrument"]).sort_index(), None),
        ("panel as D10 rebuilds it", panel_d10, None),
        ("panel with D8 year trimming", panel_d8, None)]:
        print(f"\n--- {label} ---")
        if key not in frame.index:
            print("  ROW ABSENT")
            continue
        row = frame.loc[key]
        print(row[[c for c in cols if c in frame.columns]] if cols else row)


def dump_raw_endpoint_rows(args, d10_limits, d8_limits, d10_susp, d8_susp):
    for endpoint, d10_frame, d8_frame in [("stk_limit", d10_limits, d8_limits),
                                          ("suspend_d", d10_susp, d8_susp)]:
        print(f"\n--- raw {endpoint} rows for {args.instrument} on {pd.Timestamp(args.date).date()} ---")
        for style, frame in [("D10 read_years", d10_frame), ("D8 load_yearly", d8_frame)]:
            if "ts_code" not in frame:
                print(f"  {style}: no ts_code column")
                continue
            hit = frame.loc[frame.trade_date.eq(pd.Timestamp(args.date))
                            & frame.ts_code.astype(str).str[:6].eq(args.instrument[2:])]
            print(f"  {style}: {len(hit)} row(s)")
            if len(hit):
                print(hit.to_string(index=False))

def scan_orders(access, config, window, start, end, panel_d10, args):
    print("\n## frozen orders that the rebuilt panel would reject")
    reports = Path(config["data"]["reports_dir"])
    prefix = "d8" if window == "design" else "d9_v2"
    total = 0
    for layer in range(7):
        trades = read_parquet(access, reports / f"{prefix}_layer_{layer}_trades.parquet",
                              "execution_date", start, end)
        executed = trades.loc[trades.executed_trade_amount.ne(0)].copy()
        joined = executed.join(panel_d10[["can_buy", "can_sell"]],
                               on=["execution_date", "instrument"], rsuffix="_panel")
        allowed = joined.can_buy_panel.where(joined.executed_trade_amount.gt(0), joined.can_sell_panel)
        bad = joined.loc[~allowed.fillna(False).astype(bool)]
        marker = "  <-- layers 0/1 use the ideal panel, flags not enforced there" if layer < 2 else ""
        print(f"  layer {layer}: {len(executed)} executed rows, {len(bad)} rejected{marker}")
        total += 0 if layer < 2 else len(bad)
        if len(bad) and layer >= 2:
            show = bad[["capital", "execution_date", "instrument", "executed_trade_amount",
                        "desired_trade_amount", "can_buy", "can_sell", "can_buy_panel",
                        "can_sell_panel", "blocked_by_tradability"]]
            print(show.head(args.max_rows).to_string(index=False))
            print(f"    distinct dates: {bad.execution_date.nunique()}, "
                  f"distinct instruments: {bad.instrument.nunique()}, "
                  f"date span {bad.execution_date.min().date()}..{bad.execution_date.max().date()}")
    return total


if __name__ == "__main__":
    main()

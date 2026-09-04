from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d2_core import is_close, to_qlib_symbol
from src.d3_core import (
    CostModel,
    assert_feature_timestamps,
    assert_future_price_invariance,
    build_causal_price_features,
    make_label_windows,
    make_snapshot_targets,
    make_weekly_targets,
    replication_metrics,
    run_vwap_backtest,
)
from src.data_access import ResearchDataAccess
from src.date_utils import parse_date_series


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_yearly(
    access: ResearchDataAccess,
    config: dict,
    prefix: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    frames = []
    for year in range(start_year, end_year + 1):
        relative = raw_dir / f"{prefix}_{year}.parquet"
        if not (ROOT / relative).exists():
            raise FileNotFoundError(f"缺少 D3 所需数据：{relative}")
        frames.append(
            access.read_parquet(
                relative,
                date_col="trade_date",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = parse_date_series(result["trade_date"])
    return result


def load_qlib_market(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    amount_multiplier: float,
) -> pd.DataFrame:
    fields = [
        "$open",
        "$high",
        "$low",
        "$close",
        "$vwap",
        "$volume",
        "$factor",
        "$amount",
    ]
    raw = access.features(instruments, fields, start_date, end_date)
    market = raw.reset_index().rename(columns={"datetime": "trade_date"})
    market.columns = [str(column).removeprefix("$") for column in market.columns]
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    numeric = ["open", "high", "low", "close", "vwap", "volume", "factor", "amount"]
    market[numeric] = market[numeric].apply(pd.to_numeric, errors="coerce")
    market["raw_high"] = market["high"] / market["factor"]
    market["raw_low"] = market["low"] / market["factor"]
    market["raw_close"] = market["close"] / market["factor"]
    market["daily_amount_cny"] = market["amount"] * amount_multiplier
    return market.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def add_exchange_flags(
    market: pd.DataFrame,
    limits: pd.DataFrame,
    suspensions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    result = market.copy()
    limit_rows = limits.assign(
        instrument=limits["ts_code"].map(to_qlib_symbol)
    ).drop_duplicates(["trade_date", "instrument"])
    result = result.merge(
        limit_rows[["trade_date", "instrument", "up_limit", "down_limit"]],
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    suspended = (
        suspensions.loc[suspensions["suspend_type"].eq("S")]
        .assign(instrument=lambda frame: frame["ts_code"].map(to_qlib_symbol))
        [["trade_date", "instrument"]]
        .drop_duplicates()
        .assign(has_suspend_record=True)
    )
    result = result.merge(
        suspended,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    rtol = float(config["tradability"]["one_price_relative_tolerance"])
    atol = float(config["tradability"]["one_price_absolute_tolerance_cny"])
    one_price = is_close(result["raw_high"], result["raw_low"], rtol, atol)
    result["is_one_price_limit_up"] = (
        one_price & is_close(result["raw_close"], result["up_limit"], rtol, atol)
    ).fillna(False)
    result["is_one_price_limit_down"] = (
        one_price & is_close(result["raw_close"], result["down_limit"], rtol, atol)
    ).fillna(False)
    result["is_suspended"] = (
        result["has_suspend_record"].eq(True)
        | result["close"].isna()
        | result["vwap"].isna()
    )
    result["exchange_can_buy"] = (
        ~result["is_suspended"] & ~result["is_one_price_limit_up"]
    )
    result["exchange_can_sell"] = (
        ~result["is_suspended"] & ~result["is_one_price_limit_down"]
    )
    return result.drop(columns=["has_suspend_record"])


def add_ex_ante_volatility(market: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = market.sort_values(["instrument", "trade_date"]).copy()
    impact = config["costs"]["impact"]
    lookback = int(impact["volatility_lookback_trade_days"])
    lag = int(impact["volatility_lag_trade_days"])
    minimum = int(impact["minimum_observations"])
    fallback = float(impact["fallback_daily_volatility"])
    result["return_1d"] = result.groupby("instrument", sort=False)[
        "close"
    ].pct_change(fill_method=None)
    realized = result.groupby("instrument", sort=False)["return_1d"].transform(
        lambda series: series.rolling(lookback, min_periods=minimum).std()
    )
    result["volatility"] = realized.groupby(result["instrument"]).shift(lag)
    cross_section_median = result.groupby("trade_date")["volatility"].transform(
        "median"
    )
    result["volatility"] = (
        result["volatility"].fillna(cross_section_median).fillna(fallback)
    )
    if result["volatility"].isna().any() or result["volatility"].lt(0).any():
        raise ValueError("事前波动率生成失败")
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def prepare_markets(
    market: pd.DataFrame, tradability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["trade_date", "instrument"]
    flags = tradability[keys + ["can_buy", "can_sell"]].copy()
    merged = market.merge(
        flags,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_d2"),
    )
    base_columns = [
        "trade_date",
        "instrument",
        "close",
        "vwap",
        "daily_amount_cny",
        "volatility",
    ]
    strategy = merged[base_columns].copy()
    strategy["can_buy"] = merged["can_buy"].eq(True)
    # 已调出成分的旧持仓仍必须能卖，因此卖出使用全市场交易状态。
    strategy["can_sell"] = merged["exchange_can_sell"].eq(True)
    replication = merged[base_columns].copy()
    replication["can_buy"] = merged["exchange_can_buy"].eq(True)
    replication["can_sell"] = merged["exchange_can_sell"].eq(True)
    return strategy, replication


def choose_perturbation_cutoffs(
    eligible_dates: pd.DatetimeIndex, count: int
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(eligible_dates).sort_values().unique()
    if len(dates) < 2:
        raise ValueError("未来价格扰动可用日期不足")
    candidate = dates[:-1]
    count = min(max(1, int(count)), len(candidate))
    positions = np.linspace(0, len(candidate) - 1, count, dtype=int)
    return [pd.Timestamp(candidate[position]) for position in np.unique(positions)]


def summarize_execution(trades: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    if trades.empty:
        raise ValueError("撮合没有产生任何订单")
    date_pairs = trades[["signal_date", "execution_date"]].drop_duplicates()
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    expected = date_pairs["signal_date"].map(
        lambda date: calendar[int(positions.loc[pd.Timestamp(date)]) + 1]
    )
    t_plus_one_ok = bool(
        pd.DatetimeIndex(date_pairs["execution_date"]).equals(pd.DatetimeIndex(expected))
    )
    executed = trades["executed_trade_amount"].abs()
    desired = trades["desired_trade_amount"].abs()
    return {
        "order_rows": len(trades),
        "executed_rows": int(executed.gt(0).sum()),
        "blocked_by_tradability_rows": int(trades["blocked_by_tradability"].sum()),
        "participation_limited_rows": int(trades["participation_limited"].sum()),
        "cash_limited_rows": int(trades["cash_limited"].sum()),
        "fill_ratio": float(executed.sum() / desired.sum()) if desired.sum() else 1.0,
        "t_plus_one_ok": t_plus_one_ok,
    }


def write_report(summary: dict, reports_dir: Path) -> None:
    acceptance = summary["acceptance"]
    status = "通过" if acceptance["passed"] else "未通过"
    gross = summary["replication"]["gross"]
    net = summary["replication"]["net"]
    lines = [
        "# D3 撮合、成本与防泄漏报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 防泄漏检查",
        "",
        f"- 全量时间戳断言：{'通过' if summary['leakage']['timestamp_assertion_passed'] else '未通过'}",
        f"- 未来价格扰动：{summary['leakage']['future_perturbation_passed']}/{summary['leakage']['future_perturbation_checks']} 个截止日通过",
        f"- 时间戳审计样本：{summary['leakage']['timestamp_rows']:,} 行",
        "",
        "## 撮合检查",
        "",
        f"- 信号到成交严格 T+1：{'是' if summary['execution']['t_plus_one_ok'] else '否'}",
        f"- 周频调仓信号：{summary['execution']['weekly_signal_days']:,} 个",
        f"- 订单记录：{summary['execution']['order_rows']:,} 行",
        f"- 因停牌/涨跌停等未成交：{summary['execution']['blocked_by_tradability_rows']:,} 行",
        f"- 触及 5% 成交额上限：{summary['execution']['participation_limited_rows']:,} 行",
        "",
        "## 基准复制",
        "",
        f"- 毛费前年化收益：{gross['portfolio_annualized_return']:.2%}",
        f"- H00905.CSI 年化收益：{gross['benchmark_annualized_return']:.2%}",
        f"- 毛费前年化差：{gross['annualized_difference']:.2%}",
        f"- 验收阈值：绝对年化差 < {summary['replication']['tolerance']:.2%}",
        f"- 费后年化收益：{net['portfolio_annualized_return']:.2%}",
        f"- 费后相对指数年化差：{net['annualized_difference']:.2%}",
        f"- 费后累计交易成本：{net['total_transaction_cost']:,.2f} 元",
        "",
        "## 成本口径",
        "",
        "- 佣金万 1.5、过户费万 0.1，均为双边。",
        "- 印花税仅卖出收取：2023-08-27 及以前 1‰；2023-08-28 起 0.5‰。",
        "- 冲击成本：`0.5 × 事前日波动率 × sqrt(成交金额 / 当日成交额) × 成交金额`。",
        "- 事前波动率使用滞后 1 个交易日的 20 日历史波动率；当日成交额只用于容量约束与事后冲击估算。",
        "",
        "## 验收项",
        "",
        f"- 时间戳断言：{'通过' if acceptance['timestamp_assertion'] else '未通过'}",
        f"- 未来价格扰动：{'通过' if acceptance['future_perturbation'] else '未通过'}",
        f"- T+1 撮合：{'通过' if acceptance['t_plus_one'] else '未通过'}",
        f"- 基准复制：{'通过' if acceptance['benchmark_replication'] else '未通过'}",
        "",
        "> 单元测试需在运行本管线前执行 `PYTHONPATH=. python -m pytest tests/test_d2_core.py tests/test_d3_core.py -q`；本报告只记录全量数据验收。",
    ]
    (reports_dir / "D3撮合与防泄漏报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    config = load_config()
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D3 必须在 design 模式运行，不得解封 OOS 数据")
    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    processed_dir = Path(config["data"]["processed_design_dir"])
    reports_dir = ROOT / config["data"]["reports_dir"]

    calendar_frame = access.read_parquet(
        raw_dir / "trade_cal.parquet",
        date_col="cal_date",
        start_date=design_start,
        end_date=design_end,
    )
    calendar_frame["cal_date"] = parse_date_series(calendar_frame["cal_date"])
    calendar = pd.DatetimeIndex(
        calendar_frame.loc[calendar_frame["is_open"].astype(int).eq(1), "cal_date"]
    ).sort_values()
    weights = access.read_parquet(
        processed_dir / "benchmark_weights.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    weights["trade_date"] = pd.to_datetime(weights["trade_date"]).dt.normalize()
    tradability = access.read_parquet(
        processed_dir / "tradability_flags.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    tradability["trade_date"] = pd.to_datetime(tradability["trade_date"]).dt.normalize()

    instruments = sorted(weights["instrument"].unique())
    market = load_qlib_market(
        access,
        instruments,
        design_start,
        design_end,
        float(config["tradability"]["qlib_amount_multiplier_to_cny"]),
    )
    limits = load_yearly(
        access, config, "stk_limit", design_start.year, design_end.year
    )
    suspensions = load_yearly(
        access, config, "suspend_d", design_start.year, design_end.year
    )
    market = add_exchange_flags(market, limits, suspensions, config)
    market = add_ex_ante_volatility(market, config)
    strategy_market, replication_market = prepare_markets(market, tradability)

    weekly_targets = make_weekly_targets(weights)
    horizon = int(config["label"]["horizon_trade_days"])
    label_windows = make_label_windows(
        weekly_targets["signal_date"].unique(), calendar, horizon
    )
    valid_signal_dates = pd.DatetimeIndex(label_windows["sample_date"].unique())
    weekly_targets = weekly_targets.loc[
        weekly_targets["signal_date"].isin(valid_signal_dates)
    ].copy()

    feature_builder = partial(
        build_causal_price_features,
        momentum_lookback=horizon,
        volatility_lookback=int(
            config["costs"]["impact"]["volatility_lookback_trade_days"]
        ),
    )
    audit_features = feature_builder(
        market[
            [
                "trade_date",
                "instrument",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "daily_amount_cny",
            ]
        ]
    )
    feature_metadata = (
        audit_features.loc[
            audit_features["trade_date"].isin(valid_signal_dates),
            ["trade_date", "instrument", "feature_max_timestamp"],
        ]
        .rename(columns={"trade_date": "sample_date"})
        .reset_index(drop=True)
    )
    assert_feature_timestamps(feature_metadata, label_windows)

    cutoffs = choose_perturbation_cutoffs(
        valid_signal_dates, int(config["d3"]["future_perturbation_checks"])
    )
    perturbation_passed = 0
    feature_source = market[
        [
            "trade_date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "daily_amount_cny",
        ]
    ]
    for number, cutoff in enumerate(cutoffs):
        assert_future_price_invariance(
            feature_builder,
            feature_source,
            cutoff,
            seed=int(config["project"]["random_seed"]) + number,
            rtol=float(config["d3"]["future_perturbation_rtol"]),
            atol=float(config["d3"]["future_perturbation_atol"]),
        )
        perturbation_passed += 1
        print(f"未来价格扰动：{perturbation_passed}/{len(cutoffs)}", flush=True)

    cost_model = CostModel.from_config(config)
    weekly_audits = []
    weekly_summaries = {}
    for capital in config["portfolio"]["capital_scenarios"]:
        result = run_vwap_backtest(
            strategy_market, weekly_targets, float(capital), cost_model
        )
        audit = result.trades.copy()
        audit["capital"] = float(capital)
        weekly_audits.append(audit)
        weekly_summaries[str(int(capital))] = {
            **summarize_execution(result.trades, calendar),
            "ending_nav": float(result.nav["nav"].iloc[-1]),
            "total_transaction_cost": float(result.nav["transaction_cost"].sum()),
            "mean_single_sided_turnover": float(
                result.nav["single_sided_turnover"].mean()
            ),
        }
        print(f"周频撮合完成：资金 {capital:,.0f} 元", flush=True)

    snapshot_targets = make_snapshot_targets(weights)
    first_usable = calendar[
        min(
            len(calendar) - 2,
            int(config["costs"]["impact"]["minimum_observations"])
            + int(config["costs"]["impact"]["volatility_lag_trade_days"]),
        )
    ]
    snapshot_targets = snapshot_targets.loc[
        snapshot_targets["signal_date"].ge(first_usable)
    ].copy()
    capital = float(config["portfolio"]["capital_scenarios"][0])
    gross_model = CostModel.zero_cost(cost_model.participation_limit)
    gross_replication = run_vwap_backtest(
        replication_market, snapshot_targets, capital, gross_model
    )
    net_replication = run_vwap_backtest(
        replication_market, snapshot_targets, capital, cost_model
    )
    benchmark = access.read_parquet(
        raw_dir / "index_daily_H00905_CSI.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    benchmark["trade_date"] = parse_date_series(benchmark["trade_date"])
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    benchmark = benchmark.sort_values("trade_date")
    gross_metrics, gross_yearly = replication_metrics(
        gross_replication.nav, benchmark
    )
    net_metrics, net_yearly = replication_metrics(net_replication.nav, benchmark)
    gross_execution = summarize_execution(gross_replication.trades, calendar)
    net_execution = summarize_execution(net_replication.trades, calendar)
    gross_metrics["blocked_trade_rows"] = gross_execution[
        "blocked_by_tradability_rows"
    ]
    net_metrics["blocked_trade_rows"] = net_execution[
        "blocked_by_tradability_rows"
    ]

    replication_daily = gross_replication.nav.rename(
        columns={"nav": "gross_nav"}
    )[["trade_date", "gross_nav"]].merge(
        net_replication.nav.rename(columns={"nav": "net_nav"})[
            ["trade_date", "net_nav"]
        ],
        on="trade_date",
        how="outer",
        validate="one_to_one",
    ).merge(
        benchmark[["trade_date", "close"]].rename(
            columns={"close": "benchmark_total_return_index"}
        ),
        on="trade_date",
        how="left",
        validate="one_to_one",
    )
    yearly = gross_yearly.rename(
        columns={
            "portfolio_return": "gross_portfolio_return",
            "difference": "gross_difference",
        }
    ).merge(
        net_yearly[["year", "portfolio_return", "difference"]].rename(
            columns={
                "portfolio_return": "net_portfolio_return",
                "difference": "net_difference",
            }
        ),
        on="year",
        how="outer",
        validate="one_to_one",
    )

    execution_summary = weekly_summaries[str(int(config["portfolio"]["capital_scenarios"][0]))]
    tolerance = float(config["d3"]["replication_max_annualized_difference"])
    acceptance = {
        "timestamp_assertion": True,
        "future_perturbation": perturbation_passed == len(cutoffs),
        "t_plus_one": bool(execution_summary["t_plus_one_ok"]),
        "benchmark_replication": bool(
            gross_metrics["absolute_annualized_difference"] < tolerance
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    summary = {
        "mode": access.mode,
        "window": {"start": design_start, "end": design_end},
        "leakage": {
            "timestamp_assertion_passed": True,
            "timestamp_rows": len(feature_metadata),
            "future_perturbation_checks": len(cutoffs),
            "future_perturbation_passed": int(perturbation_passed),
            "cutoffs": cutoffs,
        },
        "execution": {
            **execution_summary,
            "weekly_signal_days": int(weekly_targets["signal_date"].nunique()),
            "capital_scenarios": weekly_summaries,
        },
        "replication": {
            "capital": capital,
            "snapshot_signal_days": int(snapshot_targets["signal_date"].nunique()),
            "tolerance": tolerance,
            "gross": gross_metrics,
            "net": net_metrics,
        },
        "acceptance": acceptance,
    }

    atomic_json(summary, reports_dir / "d3_summary.json")
    write_report(summary, reports_dir)
    atomic_parquet(replication_daily, reports_dir / "d3_replication_daily.parquet")
    atomic_parquet(yearly, reports_dir / "d3_replication_yearly.parquet")
    atomic_parquet(
        gross_replication.trades,
        reports_dir / "d3_replication_trade_audit.parquet",
    )
    atomic_parquet(
        pd.concat(weekly_audits, ignore_index=True),
        reports_dir / "d3_weekly_trade_audit.parquet",
    )
    print(f"D3 验收：{'通过' if acceptance['passed'] else '未通过'}")
    print(f"报告：{reports_dir / 'D3撮合与防泄漏报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

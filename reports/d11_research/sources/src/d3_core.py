from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


class LeakageViolation(AssertionError):
    """Raised when feature lineage or perturbation reveals look-ahead."""


@dataclass(frozen=True)
class CostModel:
    commission_rate: float
    transfer_fee_rate: float
    stamp_duty: tuple[tuple[pd.Timestamp, pd.Timestamp, float], ...]
    impact_coefficient: float
    participation_limit: float

    @classmethod
    def from_config(cls, config: dict) -> CostModel:
        schedule = tuple(
            (
                pd.Timestamp(item["start_date"]).normalize(),
                pd.Timestamp(item["end_date"]).normalize(),
                float(item["rate"]),
            )
            for item in config["costs"]["stamp_duty"]
        )
        model = cls(
            commission_rate=float(config["costs"]["commission_rate"]),
            transfer_fee_rate=float(config["costs"]["transfer_fee_rate"]),
            stamp_duty=schedule,
            impact_coefficient=float(config["costs"]["impact"]["coefficient"]),
            participation_limit=float(config["execution"]["participation_limit"]),
        )
        model.validate()
        return model

    @classmethod
    def zero_cost(cls, participation_limit: float) -> CostModel:
        model = cls(
            commission_rate=0.0,
            transfer_fee_rate=0.0,
            # 不使用 Timestamp.min.normalize()：Pandas 在纳秒时间戳下可能发生
            # 边界回绕，导致起点变成 2262 年，从而无法覆盖历史回测日期。
            stamp_duty=(
                (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-12-31"), 0.0),
            ),
            impact_coefficient=0.0,
            participation_limit=float(participation_limit),
        )
        model.validate()
        return model

    def validate(self) -> None:
        rates = [
            self.commission_rate,
            self.transfer_fee_rate,
            self.impact_coefficient,
            self.participation_limit,
        ]
        if any(not np.isfinite(value) or value < 0 for value in rates):
            raise ValueError("费率、冲击系数和成交占比上限必须是非负有限数")
        if self.participation_limit > 1:
            raise ValueError("成交占比上限不能超过 1")
        ordered = sorted(self.stamp_duty, key=lambda item: item[0])
        for index, (start, end, rate) in enumerate(ordered):
            if start > end or not np.isfinite(rate) or rate < 0:
                raise ValueError("印花税区间无效")
            if index and start <= ordered[index - 1][1]:
                raise ValueError("印花税区间存在重叠")

    def stamp_rate(self, trade_date) -> float:
        date = pd.Timestamp(trade_date).normalize()
        matched = [
            rate for start, end, rate in self.stamp_duty if start <= date <= end
        ]
        if len(matched) != 1:
            raise ValueError(f"{date.date()} 未命中唯一印花税区间")
        return float(matched[0])


@dataclass
class MatchResult:
    trades: pd.DataFrame
    cash_after: float
    total_cost: float


@dataclass
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame


def trade_cost(
    signed_trade_amount: float,
    volatility: float,
    daily_amount_cny: float,
    trade_date,
    model: CostModel,
) -> dict[str, float]:
    """Calculate costs for one signed trade; buys are positive and sells negative."""
    signed = float(signed_trade_amount)
    notional = abs(signed)
    if notional == 0:
        return {
            "commission_cost": 0.0,
            "transfer_fee_cost": 0.0,
            "stamp_duty_cost": 0.0,
            "impact_cost": 0.0,
            "total_cost": 0.0,
            "impact_rate": 0.0,
        }
    volatility = float(volatility)
    daily_amount_cny = float(daily_amount_cny)
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError("非零成交必须有非负有限的波动率")
    if not np.isfinite(daily_amount_cny) or daily_amount_cny <= 0:
        raise ValueError("非零成交必须有正的当日成交额")
    participation = notional / daily_amount_cny
    if participation > model.participation_limit + 1e-12:
        raise ValueError("成交金额超过当日成交额上限")

    commission = notional * model.commission_rate
    transfer = notional * model.transfer_fee_rate
    stamp = notional * model.stamp_rate(trade_date) if signed < 0 else 0.0
    impact_rate = model.impact_coefficient * volatility * np.sqrt(participation)
    impact = notional * impact_rate
    total = commission + transfer + stamp + impact
    return {
        "commission_cost": float(commission),
        "transfer_fee_cost": float(transfer),
        "stamp_duty_cost": float(stamp),
        "impact_cost": float(impact),
        "total_cost": float(total),
        "impact_rate": float(impact_rate),
    }


def _total_buy_cash_need(
    buys: pd.Series,
    market: pd.DataFrame,
    trade_date,
    model: CostModel,
) -> float:
    total = float(buys.sum())
    for instrument, amount in buys.items():
        if amount <= 0:
            continue
        costs = trade_cost(
            amount,
            market.at[instrument, "volatility"],
            market.at[instrument, "daily_amount_cny"],
            trade_date,
            model,
        )
        total += costs["total_cost"]
    return total


def _cash_roundoff_tolerance(*scales: float) -> float:
    """Return a scale-aware tolerance for currency aggregation roundoff."""

    finite = [abs(float(value)) for value in scales if np.isfinite(value)]
    scale = max([1.0, *finite])
    return max(1e-8, 1e-12 * scale)


def match_target_orders(
    position_values: pd.Series,
    cash: float,
    target_weights: pd.Series,
    market: pd.DataFrame,
    trade_date,
    model: CostModel,
) -> MatchResult:
    """Match one rebalance at T+1 VWAP with exchange and cash constraints."""
    required_columns = {
        "execution_price",
        "daily_amount_cny",
        "volatility",
        "can_buy",
        "can_sell",
    }
    missing = required_columns - set(market.columns)
    if missing:
        raise ValueError(f"撮合行情缺少字段：{sorted(missing)}")

    positions = pd.to_numeric(position_values, errors="raise").astype(float)
    targets = pd.to_numeric(target_weights, errors="raise").astype(float)
    if (positions < -1e-8).any() or (targets < -1e-12).any():
        raise ValueError("本轮只允许多头持仓和非负目标权重")
    if targets.sum() > 1.0 + 1e-8:
        raise ValueError("目标权重和不能超过 1")
    cash = float(cash)
    input_cash_tolerance = _cash_roundoff_tolerance(
        cash,
        float(positions.abs().sum()),
    )
    if not np.isfinite(cash) or cash < -input_cash_tolerance:
        raise ValueError("现金必须是非负有限数")
    # Hundreds of notionals and nonlinear impact costs are aggregated in
    # float64. A sub-cent negative residue is roundoff, not borrowing.
    if cash < 0:
        cash = 0.0

    universe = positions.index.union(targets.index).union(market.index)
    positions = positions.reindex(universe, fill_value=0.0)
    targets = targets.reindex(universe, fill_value=0.0)
    quotes = market.reindex(universe).copy()
    quotes["can_buy"] = quotes["can_buy"].eq(True)
    quotes["can_sell"] = quotes["can_sell"].eq(True)
    # Qlib may return float32 amounts. Computing a 5% cap in float32 and then
    # validating it in float64 can make the clipped notional appear above the
    # cap by several cents. Normalize cost inputs before deriving capacity.
    quotes["daily_amount_cny"] = pd.to_numeric(
        quotes["daily_amount_cny"], errors="coerce"
    ).astype("float64")
    quotes["volatility"] = pd.to_numeric(
        quotes["volatility"], errors="coerce"
    ).astype("float64")
    daily_amount = quotes["daily_amount_cny"]
    capacity = (daily_amount * model.participation_limit).where(
        daily_amount.gt(0), 0.0
    )

    nav = float(positions.sum() + cash)
    if nav <= 0:
        raise ValueError("撮合前净值必须为正")
    desired = targets * nav - positions
    sells = (-desired.clip(upper=0.0)).clip(upper=positions)
    executable_sells = sells.where(quotes["can_sell"], 0.0).clip(upper=capacity)
    signed_sells = -executable_sells

    sell_cost = 0.0
    for instrument, amount in signed_sells.items():
        if amount == 0:
            continue
        sell_cost += trade_cost(
            amount,
            quotes.at[instrument, "volatility"],
            quotes.at[instrument, "daily_amount_cny"],
            trade_date,
            model,
        )["total_cost"]
    cash_after_sells = float(cash + executable_sells.sum() - sell_cost)
    cash_after_sells_tolerance = _cash_roundoff_tolerance(
        nav,
        cash_after_sells,
        float(executable_sells.sum()),
        sell_cost,
    )
    if cash_after_sells < -cash_after_sells_tolerance:
        raise ValueError("卖出后现金为负，费用或持仓数据异常")
    if cash_after_sells < 0:
        cash_after_sells = 0.0

    buys = desired.clip(lower=0.0)
    candidate_buys = buys.where(quotes["can_buy"], 0.0).clip(upper=capacity)
    if _total_buy_cash_need(candidate_buys, quotes, trade_date, model) <= cash_after_sells:
        executable_buys = candidate_buys
        cash_scale = 1.0
    elif candidate_buys.sum() <= 0 or cash_after_sells <= 0:
        executable_buys = candidate_buys * 0.0
        cash_scale = 0.0
    else:
        lower, upper = 0.0, 1.0
        for _ in range(80):
            middle = (lower + upper) / 2
            needed = _total_buy_cash_need(
                candidate_buys * middle, quotes, trade_date, model
            )
            if needed <= cash_after_sells:
                lower = middle
            else:
                upper = middle
        cash_scale = lower
        executable_buys = candidate_buys * cash_scale

    executed = executable_buys + signed_sells
    records = []
    total_cost = 0.0
    for instrument in universe:
        amount = float(executed.loc[instrument])
        if amount:
            costs = trade_cost(
                amount,
                quotes.at[instrument, "volatility"],
                quotes.at[instrument, "daily_amount_cny"],
                trade_date,
                model,
            )
        else:
            costs = trade_cost(0.0, 0.0, 1.0, trade_date, model)
        total_cost += costs["total_cost"]
        desired_amount = float(desired.loc[instrument])
        side_allowed = (
            bool(quotes.at[instrument, "can_buy"])
            if desired_amount > 0
            else bool(quotes.at[instrument, "can_sell"])
            if desired_amount < 0
            else True
        )
        participation_limited = bool(
            abs(desired_amount) > float(capacity.loc[instrument]) + 1e-8
            and side_allowed
        )
        cash_limited = bool(
            desired_amount > 0
            and candidate_buys.loc[instrument] > 0
            and cash_scale < 1.0 - 1e-10
        )
        records.append(
            {
                "instrument": instrument,
                "target_weight": float(targets.loc[instrument]),
                "pretrade_weight": float(positions.loc[instrument] / nav),
                "desired_trade_amount": desired_amount,
                "executed_trade_amount": amount,
                "can_buy": bool(quotes.at[instrument, "can_buy"]),
                "can_sell": bool(quotes.at[instrument, "can_sell"]),
                "participation_limited": participation_limited,
                "cash_limited": cash_limited,
                "blocked_by_tradability": bool(desired_amount and not side_allowed),
                **costs,
            }
        )

    buy_notional = float(executable_buys.sum())
    sell_notional = float(executable_sells.sum())
    cash_after = float(cash + sell_notional - buy_notional - total_cost)
    output_cash_tolerance = _cash_roundoff_tolerance(
        nav,
        buy_notional,
        sell_notional,
        total_cost,
    )
    if cash_after < -output_cash_tolerance:
        raise AssertionError(f"撮合后现金为负：{cash_after}")
    if cash_after < 0 or abs(cash_after) < 1e-8:
        cash_after = 0.0
    return MatchResult(pd.DataFrame(records), cash_after, float(total_cost))


def _validate_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "instrument",
        "close",
        "vwap",
        "daily_amount_cny",
        "volatility",
        "can_buy",
        "can_sell",
    }
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"回测行情缺少字段：{sorted(missing)}")
    result = market.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    if result.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("回测行情存在重复的 日期×股票")
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def _coalesce_numeric_series(
    primary: pd.Series, *fallbacks: pd.Series
) -> pd.Series:
    """Fill missing numeric values without Pandas' empty-concat dtype ambiguity."""
    result = pd.to_numeric(primary, errors="coerce").astype("float64").copy()
    for fallback in fallbacks:
        candidate = pd.to_numeric(
            fallback.reindex(result.index), errors="coerce"
        ).astype("float64")
        missing = result.isna() & candidate.notna()
        if missing.any():
            result.loc[missing] = candidate.loc[missing]
    return result


def run_vwap_backtest(
    market: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float,
    model: CostModel,
) -> BacktestResult:
    """Run a long-only daily NAV simulation; every signal trades next market day."""
    prices = _validate_market(market)
    targets = target_weights.copy()
    required_targets = {"signal_date", "instrument", "target_weight"}
    missing = required_targets - set(targets.columns)
    if missing:
        raise ValueError(f"目标权重缺少字段：{sorted(missing)}")
    targets["signal_date"] = pd.to_datetime(targets["signal_date"]).dt.normalize()
    if targets.duplicated(["signal_date", "instrument"]).any():
        raise ValueError("目标权重存在重复的 信号日×股票")
    target_sums = targets.groupby("signal_date")["target_weight"].sum()
    if (target_sums > 1.0 + 1e-8).any() or (targets["target_weight"] < 0).any():
        raise ValueError("目标权重必须非负且每日权重和不超过 1")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("初始资金必须为正")

    calendar = pd.DatetimeIndex(prices["trade_date"].unique()).sort_values()
    execution_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in pd.DatetimeIndex(targets["signal_date"].unique()).sort_values():
        later = calendar[calendar > signal_date]
        if len(later):
            execution_date = pd.Timestamp(later[0])
            if execution_date in execution_map:
                raise ValueError("多个信号映射到同一个执行日")
            execution_map[execution_date] = pd.Timestamp(signal_date)
    grouped_targets = {
        pd.Timestamp(date): group.set_index("instrument")["target_weight"].astype(float)
        for date, group in targets.groupby("signal_date")
    }
    market_by_date = {
        pd.Timestamp(date): group.set_index("instrument")
        for date, group in prices.groupby("trade_date", sort=True)
    }

    cash = float(initial_capital)
    units = pd.Series(dtype=float)
    last_price = pd.Series(dtype=float)
    nav_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    for date in calendar:
        day = market_by_date[pd.Timestamp(date)]
        close_today = pd.to_numeric(day["close"], errors="coerce").where(
            lambda series: series.gt(0)
        )
        vwap_today = pd.to_numeric(day["vwap"], errors="coerce").where(
            lambda series: series.gt(0)
        )
        daily_cost = 0.0
        turnover = 0.0
        traded_names = 0
        signal_date = execution_map.get(pd.Timestamp(date))

        if signal_date is not None:
            target = grouped_targets[signal_date]
            universe = units.index.union(target.index)
            execution_price = _coalesce_numeric_series(
                vwap_today.reindex(universe),
                close_today.reindex(universe),
                last_price.reindex(universe),
            )
            held = units.reindex(universe, fill_value=0.0)
            if execution_price.loc[held.gt(0)].isna().any():
                raise ValueError(f"{date.date()} 无法给已有持仓估值")
            position_values = held * execution_price.fillna(0.0)
            quotes = day.reindex(universe).copy()
            quotes["execution_price"] = execution_price
            valid_execution = execution_price.notna() & execution_price.gt(0)
            quotes["can_buy"] = quotes["can_buy"].eq(True) & valid_execution
            quotes["can_sell"] = quotes["can_sell"].eq(True) & valid_execution
            result = match_target_orders(
                position_values,
                cash,
                target,
                quotes,
                date,
                model,
            )
            trades = result.trades.loc[
                result.trades["executed_trade_amount"].ne(0)
                | result.trades["desired_trade_amount"].ne(0)
            ].copy()
            trades["signal_date"] = signal_date
            trades["execution_date"] = pd.Timestamp(date)
            if not trades.empty:
                trade_frames.append(trades)
            executed = result.trades.set_index("instrument")["executed_trade_amount"]
            nonzero = executed.loc[executed.ne(0)]
            if not nonzero.empty:
                unit_changes = nonzero / execution_price.reindex(nonzero.index)
                units = units.reindex(units.index.union(unit_changes.index), fill_value=0.0)
                units.loc[unit_changes.index] = units.loc[unit_changes.index] + unit_changes
                units = units.where(units.abs().gt(1e-12)).dropna()
            pretrade_nav = float(position_values.sum() + cash)
            turnover = (
                0.5 * float(nonzero.abs().sum()) / pretrade_nav
                if pretrade_nav > 0
                else 0.0
            )
            traded_names = len(nonzero)
            cash = result.cash_after
            daily_cost = result.total_cost

        held_close = _coalesce_numeric_series(
            close_today.reindex(units.index),
            vwap_today.reindex(units.index),
            last_price.reindex(units.index),
        )
        if held_close.isna().any():
            raise ValueError(f"{date.date()} 无法计算日终持仓净值")
        holdings_value = float((units * held_close).sum())
        nav = cash + holdings_value
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"{date.date()} 组合净值无效：{nav}")
        nav_rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "nav": float(nav),
                "cash": float(cash),
                "gross_exposure": float(holdings_value / nav),
                "single_sided_turnover": float(turnover),
                "transaction_cost": float(daily_cost),
                "traded_names": traded_names,
                "signal_date": signal_date,
            }
        )
        valid_closes = close_today.dropna()
        last_price = last_price.reindex(last_price.index.union(valid_closes.index))
        last_price.loc[valid_closes.index] = valid_closes

    trade_columns = [
        "instrument",
        "target_weight",
        "pretrade_weight",
        "desired_trade_amount",
        "executed_trade_amount",
        "can_buy",
        "can_sell",
        "participation_limited",
        "cash_limited",
        "blocked_by_tradability",
        "commission_cost",
        "transfer_fee_cost",
        "stamp_duty_cost",
        "impact_cost",
        "total_cost",
        "impact_rate",
        "signal_date",
        "execution_date",
    ]
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=trade_columns)
    )
    return BacktestResult(pd.DataFrame(nav_rows), trades)


def build_causal_price_features(
    prices: pd.DataFrame,
    *,
    momentum_lookback: int = 5,
    volatility_lookback: int = 20,
) -> pd.DataFrame:
    """Small causal feature set used to audit the D3 feature-generation contract."""
    required = [
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
    missing = set(required) - set(prices.columns)
    if missing:
        raise ValueError(f"特征行情缺少字段：{sorted(missing)}")
    result = prices[required].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(["instrument", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("instrument", sort=False)
    result["return_1d"] = grouped["close"].pct_change(fill_method=None)
    result["momentum"] = result["close"] / grouped["close"].shift(momentum_lookback) - 1
    result["volatility"] = result.groupby("instrument", sort=False)[
        "return_1d"
    ].transform(
        lambda series: series.rolling(
            volatility_lookback, min_periods=volatility_lookback
        ).std()
    )
    result["vwap_close_gap"] = result["vwap"] / result["close"] - 1
    result["open_close_gap"] = result["close"] / result["open"] - 1
    result["intraday_range"] = (result["high"] - result["low"]) / result["close"]
    result["volume_mean"] = grouped["volume"].transform(
        lambda series: series.rolling(
            volatility_lookback, min_periods=volatility_lookback
        ).mean()
    )
    result["feature_max_timestamp"] = result["trade_date"]
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def make_label_windows(
    sample_dates: Iterable[pd.Timestamp],
    calendar: Iterable[pd.Timestamp],
    horizon_trade_days: int,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values()
    positions = pd.Series(np.arange(len(dates)), index=dates)
    rows = []
    for sample_date in pd.DatetimeIndex(pd.to_datetime(list(sample_dates))).normalize().unique():
        if sample_date not in positions.index:
            raise ValueError(f"样本日 {sample_date.date()} 不在交易日历")
        position = int(positions.loc[sample_date])
        exit_position = position + int(horizon_trade_days) + 1
        if exit_position >= len(dates):
            continue
        rows.append(
            {
                "sample_date": sample_date,
                "label_start_timestamp": dates[position + 1],
                "label_end_timestamp": dates[exit_position],
            }
        )
    return pd.DataFrame(rows)


def assert_feature_timestamps(
    feature_metadata: pd.DataFrame, label_windows: pd.DataFrame
) -> None:
    required_features = {"sample_date", "instrument", "feature_max_timestamp"}
    required_labels = {
        "sample_date",
        "label_start_timestamp",
        "label_end_timestamp",
    }
    if required_features - set(feature_metadata.columns):
        raise ValueError("特征时间戳元数据字段不完整")
    if required_labels - set(label_windows.columns):
        raise ValueError("标签时间窗字段不完整")
    features = feature_metadata.copy()
    labels = label_windows.copy()
    for column in ["sample_date", "feature_max_timestamp"]:
        features[column] = pd.to_datetime(features[column]).dt.normalize()
    for column in ["sample_date", "label_start_timestamp", "label_end_timestamp"]:
        labels[column] = pd.to_datetime(labels[column]).dt.normalize()
    merged = features.merge(labels, on="sample_date", how="left", validate="many_to_one")
    if merged["label_start_timestamp"].isna().any():
        raise ValueError("部分样本没有标签时间窗")
    violations = merged.loc[
        merged["feature_max_timestamp"].ge(merged["label_start_timestamp"])
    ]
    if not violations.empty:
        first = violations.iloc[0]
        raise LeakageViolation(
            "特征时间戳越过标签起点："
            f"{first['instrument']} sample={first['sample_date'].date()} "
            f"feature_max={first['feature_max_timestamp'].date()} "
            f"label_start={first['label_start_timestamp'].date()}"
        )


def assert_future_price_invariance(
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    cutoff,
    *,
    price_columns: tuple[str, ...] = ("open", "high", "low", "close", "vwap"),
    seed: int = 42,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    cutoff = pd.Timestamp(cutoff).normalize()
    source = prices.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    # Qlib 的价格列通常是 float32。若直接把 float64 扰动值写回，Pandas 会
    # 改变整列精度，使截止日前的特征仅因舍入差异而被误判为前视。
    for column in price_columns:
        if column not in source:
            raise ValueError(f"扰动价格字段不存在：{column}")
        source[column] = pd.to_numeric(source[column], errors="coerce").astype(
            "float64"
        )
    future_mask = source["trade_date"].gt(cutoff)
    if not future_mask.any():
        raise ValueError("扰动截止日之后没有价格，测试无效")
    baseline = feature_builder(source)
    rng = np.random.default_rng(seed)
    multiplier = rng.uniform(0.35, 2.75, size=int(future_mask.sum()))
    perturbed = source.copy()
    for column in price_columns:
        values = pd.to_numeric(perturbed.loc[future_mask, column], errors="coerce")
        perturbed.loc[future_mask, column] = values.to_numpy() * multiplier
    rebuilt = feature_builder(perturbed)

    keys = ["trade_date", "instrument"]
    excluded = set(keys) | {"feature_max_timestamp"}
    feature_columns = [column for column in baseline.columns if column not in excluded]
    left = (
        baseline.loc[baseline["trade_date"].le(cutoff), keys + feature_columns]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    right = (
        rebuilt.loc[rebuilt["trade_date"].le(cutoff), keys + feature_columns]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as exc:
        raise LeakageViolation(f"未来价格扰动改变了 {cutoff.date()} 以前的特征") from exc


def make_weekly_targets(daily_weights: pd.DataFrame) -> pd.DataFrame:
    weights = daily_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"]).dt.normalize()
    week = weights["trade_date"].dt.to_period("W-FRI")
    last_dates = weights.groupby(week)["trade_date"].transform("max")
    result = weights.loc[weights["trade_date"].eq(last_dates), [
        "trade_date",
        "instrument",
        "weight",
    ]].rename(columns={"trade_date": "signal_date", "weight": "target_weight"})
    return result.sort_values(["signal_date", "instrument"]).reset_index(drop=True)


def make_snapshot_targets(daily_weights: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "instrument", "weight", "snapshot_date"}
    missing = required - set(daily_weights.columns)
    if missing:
        raise ValueError(f"基准权重缺少字段：{sorted(missing)}")
    weights = daily_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"]).dt.normalize()
    weights["snapshot_date"] = pd.to_datetime(weights["snapshot_date"]).dt.normalize()
    meta = weights.groupby("trade_date")["snapshot_date"].agg(["first", "nunique"])
    if meta["nunique"].ne(1).any():
        raise ValueError("同一交易日存在多个生效快照日期")
    signal_dates = meta.index[meta["first"].ne(meta["first"].shift())]
    result = weights.loc[weights["trade_date"].isin(signal_dates), [
        "trade_date",
        "instrument",
        "weight",
    ]].rename(columns={"trade_date": "signal_date", "weight": "target_weight"})
    return result.sort_values(["signal_date", "instrument"]).reset_index(drop=True)


def replication_metrics(
    nav: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame]:
    portfolio = nav.copy()
    index = benchmark.copy()
    portfolio["trade_date"] = pd.to_datetime(portfolio["trade_date"]).dt.normalize()
    index["trade_date"] = pd.to_datetime(index["trade_date"]).dt.normalize()
    merged = portfolio.merge(
        index[["trade_date", "close"]], on="trade_date", how="inner", validate="one_to_one"
    ).sort_values("trade_date")
    invested_positions = np.flatnonzero(merged["gross_exposure"].gt(0).to_numpy())
    if len(invested_positions) == 0:
        raise ValueError("复制组合从未建仓")
    # 复制测试检验的是“建仓完成后”持有 w_b 能否跟踪全收益指数。
    # 若从首次执行日前一日开始计量，会把现金在 T+1 VWAP 建仓前漏掉的
    # 当日收益当成长期复制误差；结果会高度依赖任意的启动日期。
    start_position = int(invested_positions[0])
    merged = merged.iloc[start_position:].copy().reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError("复制组合与指数的重叠区间不足")
    merged["portfolio_return"] = merged["nav"].pct_change(fill_method=None)
    merged["benchmark_return"] = merged["close"].pct_change(fill_method=None)
    merged["active_return"] = merged["portfolio_return"] - merged["benchmark_return"]
    periods = len(merged) - 1
    portfolio_annualized = float(
        (merged["nav"].iloc[-1] / merged["nav"].iloc[0]) ** (252.0 / periods) - 1
    )
    benchmark_annualized = float(
        (merged["close"].iloc[-1] / merged["close"].iloc[0]) ** (252.0 / periods) - 1
    )
    yearly_rows = []
    for year, group in merged.groupby(merged["trade_date"].dt.year):
        portfolio_return = float((1 + group["portfolio_return"].dropna()).prod() - 1)
        benchmark_return = float((1 + group["benchmark_return"].dropna()).prod() - 1)
        yearly_rows.append(
            {
                "year": int(year),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "difference": portfolio_return - benchmark_return,
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    metrics = {
        "portfolio_annualized_return": portfolio_annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "annualized_difference": portfolio_annualized - benchmark_annualized,
        "absolute_annualized_difference": abs(
            portfolio_annualized - benchmark_annualized
        ),
        "tracking_error": float(merged["active_return"].std(ddof=1) * np.sqrt(252)),
        "mean_single_sided_turnover": float(merged["single_sided_turnover"].mean()),
        "total_transaction_cost": float(merged["transaction_cost"].sum()),
        "blocked_trade_rows": 0.0,
    }
    return metrics, yearly

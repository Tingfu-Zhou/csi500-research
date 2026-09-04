from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d3_pipeline import (  # noqa: E402
    add_exchange_flags,
    add_ex_ante_volatility,
    load_qlib_market,
    load_yearly,
    prepare_markets,
    summarize_execution,
)
from scripts.d7_pipeline import (  # noqa: E402
    build_execution_price_matrix,
    holding_returns_between,
    make_execution_schedule,
)
from src.d3_core import CostModel, run_vwap_backtest  # noqa: E402
from src.d6_core import estimate_ledoit_wolf_covariance  # noqa: E402
from src.d7_core import (  # noqa: E402
    OptimizationLimits,
    PortfolioOptimizationError,
    annualized_tracking_error,
    drift_pretrade_weights,
    optimize_long_only_active_portfolio,
    symmetric_l1_cost_rate,
)
from src.d8_core import (  # noqa: E402
    LAYER_EXPLANATIONS,
    LAYER_NAMES,
    assert_layer6_matches_d7,
    audit_target_exposures,
    build_loss_deltas,
    make_ideal_execution_market,
    make_top_n_equal_targets,
    require_d7_structural_completion,
    summarize_backtest_performance,
    validate_layer_ids,
)
from src.data_access import ResearchDataAccess  # noqa: E402
from src.date_utils import parse_date_series  # noqa: E402


def load_config() -> dict:
    with (ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_json(path: Path, missing_message: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(missing_message)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict) -> tuple[tuple[int, ...], float]:
    d8 = config["d8"]
    layer_ids = validate_layer_ids(d8["layer_ids"])
    if int(d8["top_n"]) != 100:
        raise ValueError("D8 第0层必须固定为 Top100")
    capital_scenarios = [float(value) for value in config["portfolio"]["capital_scenarios"]]
    main_capital = float(d8["main_capital_cny"])
    if main_capital not in capital_scenarios:
        raise ValueError("d8.main_capital_cny 必须来自 portfolio.capital_scenarios")
    if not np.isclose(float(d8["ideal_participation_limit"]), 1.0):
        raise ValueError("D8 理想执行层的成交占比上限必须固定为 1.0")
    if not np.isclose(
        float(d8["return_annualization_factor"]), 252.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("D8 日频收益年化因子必须固定为 252")
    return layer_ids, main_capital


def load_design_parquet(
    access: ResearchDataAccess,
    relative_path: Path,
    *,
    date_col: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    frame = access.read_parquet(
        relative_path,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    frame[date_col] = parse_date_series(frame[date_col])
    return frame


def optimizer_limits_for_layer(config: dict, layer: int) -> tuple[OptimizationLimits, list[str]]:
    d7 = config["d7"]
    tolerance = float(d7["constraint_tolerance"])
    industries_on = layer >= 3
    styles_on = layer >= 4
    turnover_on = layer >= 5
    stock_on = layer >= 6
    limits = OptimizationLimits(
        stock_active=float(d7["stock_active_limit"]) if stock_on else 1.0,
        industry_active=float(d7["industry_active_limit"]) if industries_on else 1.0,
        style_active=float(d7["style_active_limit"]) if styles_on else 1.0,
        single_sided_turnover=(
            float(d7["single_sided_turnover_limit"]) if turnover_on else 1.0
        ),
        tolerance=tolerance,
    )
    style_columns = (
        [f"style_{name}" for name in d7["constrained_style_factors"]]
        if styles_on
        else []
    )
    return limits, style_columns


def build_frozen_d7_pretrade_path(
    d7_targets: pd.DataFrame,
    exposures: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    execution_by_signal: pd.Series,
    price_matrix: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[pd.Timestamp, pd.Series]:
    """Reconstruct the exact causal pretrade path used by frozen D7.

    D8 layers 3–6 are marginal constraint ablations, so they must share one
    pretrade reference.  Using an independent path for layer 5 can make its
    15% turnover set empty after earlier unconstrained concentration.
    """

    targets = d7_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"]).dt.normalize()
    previous_target: pd.Series | None = None
    previous_execution_date: pd.Timestamp | None = None
    result: dict[pd.Timestamp, pd.Series] = {}
    for signal_date in signal_dates:
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross_section = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross_section["benchmark_weight"], errors="coerce")
        benchmark = benchmark / benchmark.sum()
        current_target = (
            targets.loc[targets["signal_date"].eq(signal_date)]
            .set_index("instrument")["target_weight"]
            .astype(float)
            .reindex(cross_section.index, fill_value=0.0)
        )
        if previous_target is None:
            pretrade = benchmark.copy()
        else:
            held = previous_target[previous_target > float(tolerance) * 0.1].index
            returns = holding_returns_between(
                price_matrix,
                previous_execution_date,
                execution_date,
                held,
            )
            pretrade = drift_pretrade_weights(
                previous_target,
                returns,
                weight_tolerance=1e-8,
            )
        result[signal_date] = pretrade
        previous_target = current_target
        previous_execution_date = execution_date
    return result


def generate_optimized_layer_targets(
    layer: int,
    predictions: pd.DataFrame,
    exposures: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    execution_by_signal: pd.Series,
    pretrade_by_signal: dict[pd.Timestamp, pd.Series],
    industry_columns: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if layer not in {3, 4, 5}:
        raise ValueError("只允许为 D8 第3–5层新生成优化目标")
    expected_return_column = str(config["d5"]["expected_return_column"])
    limits, constrained_styles = optimizer_limits_for_layer(config, layer)
    constrained_industries = industry_columns if layer >= 3 else []
    cost_model = CostModel.from_config(config)
    target_frames: list[pd.DataFrame] = []
    solver_rows: list[dict[str, object]] = []

    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross_section = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross_section["benchmark_weight"], errors="coerce")
        if benchmark.isna().any() or (benchmark < 0).any() or benchmark.sum() <= 0:
            raise ValueError(f"{signal_date.date()} 基准权重无效")
        benchmark = benchmark / benchmark.sum()
        observed = predictions.loc[predictions["sample_date"].eq(signal_date)]
        mu_observed = observed.set_index("instrument")[expected_return_column]
        unexpected = sorted(set(mu_observed.index) - set(cross_section.index))
        if unexpected:
            raise ValueError(
                f"{signal_date.date()} D5 含非当日成分：" + ", ".join(unexpected[:20])
            )
        mu = pd.to_numeric(mu_observed, errors="coerce").reindex(cross_section.index)
        if mu.loc[mu.index.isin(mu_observed.index)].isna().any():
            raise ValueError(f"{signal_date.date()} D5 已输出 μ 含非有限值")
        mu = mu.fillna(0.0).astype(float)
        pretrade = pretrade_by_signal[signal_date]
        cost_rate = symmetric_l1_cost_rate(
            cost_model.commission_rate,
            cost_model.transfer_fee_rate,
            cost_model.stamp_rate(execution_date),
        )
        try:
            result = optimize_long_only_active_portfolio(
                mu,
                benchmark,
                pretrade,
                cross_section,
                constrained_industries,
                constrained_styles,
                cost_rate=cost_rate,
                limits=limits,
                solver=str(config["d7"]["solver"]),
            )
        except PortfolioOptimizationError as exc:
            forced_exit = float(
                pretrade.loc[~pretrade.index.isin(cross_section.index)].sum()
            )
            raise PortfolioOptimizationError(
                f"D8 第{layer}层 {signal_date.date()} 求解失败；"
                f"冻结D7事前路径强制退出权重={forced_exit:.2%}；{exc}"
            ) from exc
        target_frames.append(
            pd.DataFrame(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "instrument": result.weights.index,
                    "target_weight": result.weights.to_numpy(float),
                }
            )
        )
        solver_rows.append(
            {
                "layer": layer,
                "signal_date": signal_date,
                "solver_status": result.status,
                "fallback_used": False,
                "objective_value": result.objective_value,
                "estimated_linear_cost": result.estimated_linear_cost,
                "optimizer_single_sided_turnover": result.audit[
                    "single_sided_turnover"
                ],
            }
        )
        if number == 1 or number % 20 == 0 or number == len(signal_dates):
            print(
                f"D8 第{layer}层目标 {number}/{len(signal_dates)}："
                f"{signal_date.date()}",
                flush=True,
            )
    return (
        pd.concat(target_frames, ignore_index=True),
        pd.DataFrame(solver_rows),
    )


def audit_all_layer_targets(
    layer_targets: dict[int, pd.DataFrame],
    exposures: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    execution_by_signal: pd.Series,
    price_matrix: pd.DataFrame,
    frozen_pretrade_by_signal: dict[pd.Timestamp, pd.Series],
    industry_columns: list[str],
    style_columns: list[str],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    for layer in range(7):
        targets = layer_targets[layer].copy()
        targets["signal_date"] = pd.to_datetime(targets["signal_date"]).dt.normalize()
        previous_target: pd.Series | None = None
        previous_execution_date: pd.Timestamp | None = None
        for signal_date in signal_dates:
            signal_date = pd.Timestamp(signal_date).normalize()
            execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
            cross_section = (
                exposures.loc[exposures["trade_date"].eq(signal_date)]
                .set_index("instrument")
                .sort_index()
            )
            benchmark = pd.to_numeric(cross_section["benchmark_weight"], errors="coerce")
            benchmark = benchmark / benchmark.sum()
            target = (
                targets.loc[targets["signal_date"].eq(signal_date)]
                .set_index("instrument")["target_weight"]
                .astype(float)
            )
            target = target.reindex(benchmark.index, fill_value=0.0)
            if layer >= 3:
                pretrade = frozen_pretrade_by_signal[signal_date]
                turnover_reference = "frozen_d7_pretrade"
            elif previous_target is None:
                pretrade = benchmark.copy()
                turnover_reference = "layer_own_pretrade"
            else:
                held = previous_target[previous_target > tolerance * 0.1].index
                returns = holding_returns_between(
                    price_matrix,
                    previous_execution_date,
                    execution_date,
                    held,
                )
                pretrade = drift_pretrade_weights(
                    previous_target, returns, weight_tolerance=1e-8
                )
                turnover_reference = "layer_own_pretrade"
            current_pretrade = pretrade.reindex(benchmark.index, fill_value=0.0)
            forced_exit = float(pretrade.loc[~pretrade.index.isin(benchmark.index)].sum())
            total_l1 = float((target - current_pretrade).abs().sum() + forced_exit)
            exposure_audit = audit_target_exposures(
                target,
                benchmark,
                cross_section,
                industry_columns,
                style_columns,
            )
            audit_rows.append(
                {
                    "layer": layer,
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "target_single_sided_turnover": 0.5 * total_l1,
                    "turnover_reference": turnover_reference,
                    "forced_exit_weight": forced_exit,
                    "maximum_stock_active": exposure_audit.maximum_stock_active,
                    "maximum_industry_active": exposure_audit.maximum_industry_active,
                    "maximum_style_active": exposure_audit.maximum_style_active,
                    "holdings": int((target > tolerance).sum()),
                }
            )
            for factor_type, values in [
                ("industry", exposure_audit.industry_active_exposures),
                ("style", exposure_audit.style_active_exposures),
            ]:
                factor_rows.extend(
                    {
                        "layer": layer,
                        "signal_date": signal_date,
                        "factor": factor,
                        "factor_type": factor_type,
                        "active_exposure": value,
                    }
                    for factor, value in values.items()
                )
            previous_target = target
            previous_execution_date = execution_date
    return pd.DataFrame(audit_rows), pd.DataFrame(factor_rows)


def calculate_layer_tracking_error(
    layer_targets: dict[int, pd.DataFrame],
    exposures: pd.DataFrame,
    stock_returns: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.DataFrame:
    d6 = config["d6"]
    annualization = float(config["d7"]["covariance_annualization_factor"])
    rows: list[dict[str, object]] = []
    indexed_targets = {
        layer: frame.assign(
            signal_date=pd.to_datetime(frame["signal_date"]).dt.normalize()
        )
        for layer, frame in layer_targets.items()
    }
    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        cross_section = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross_section["benchmark_weight"], errors="coerce")
        benchmark = benchmark / benchmark.sum()
        covariance = estimate_ledoit_wolf_covariance(
            stock_returns,
            as_of_date=signal_date,
            instruments=benchmark.index,
            window=int(d6["covariance_window_trade_days"]),
            minimum_observations=int(d6["covariance_minimum_observations"]),
        )
        for layer in range(7):
            target = (
                indexed_targets[layer]
                .loc[indexed_targets[layer]["signal_date"].eq(signal_date)]
                .set_index("instrument")["target_weight"]
                .astype(float)
                .reindex(benchmark.index, fill_value=0.0)
            )
            estimate = annualized_tracking_error(
                target - benchmark,
                covariance.covariance,
                covariance.instruments,
                annualization_factor=annualization,
            )
            rows.append(
                {
                    "layer": layer,
                    "signal_date": signal_date,
                    "annualized_ex_ante_tracking_error": estimate.annualized_tracking_error,
                    "covariance_active_l1_coverage": estimate.active_l1_coverage,
                    "covariance_instruments": estimate.included_instruments,
                    "covariance_shrinkage": covariance.shrinkage,
                }
            )
        if number == 1 or number % 20 == 0 or number == len(signal_dates):
            print(
                f"D8 事前 TE {number}/{len(signal_dates)}：{signal_date.date()}",
                flush=True,
            )
    return pd.DataFrame(rows)


def build_stock_returns(market: pd.DataFrame) -> pd.DataFrame:
    return (
        market.pivot(index="trade_date", columns="instrument", values="close")
        .pct_change(fill_method=None)
        .stack(future_stack=True)
        .rename("return")
        .reset_index()
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def run_loss_ladder_backtests(
    layer_targets: dict[int, pd.DataFrame],
    unconstrained_market: pd.DataFrame,
    ideal_cost_market: pd.DataFrame,
    strategy_market: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    base_cost_model = CostModel.from_config(config)
    ideal_cost_model = CostModel(
        commission_rate=base_cost_model.commission_rate,
        transfer_fee_rate=base_cost_model.transfer_fee_rate,
        stamp_duty=base_cost_model.stamp_duty,
        impact_coefficient=base_cost_model.impact_coefficient,
        participation_limit=float(config["d8"]["ideal_participation_limit"]),
    )
    ideal_cost_model.validate()
    zero_model = CostModel.zero_cost(
        participation_limit=float(config["d8"]["ideal_participation_limit"])
    )
    summary_rows: list[dict[str, object]] = []
    daily_by_layer: dict[int, list[pd.DataFrame]] = {layer: [] for layer in range(7)}
    trades_by_layer: dict[int, list[pd.DataFrame]] = {layer: [] for layer in range(7)}
    annualization = float(config["d8"]["return_annualization_factor"])

    for capital in [float(value) for value in config["portfolio"]["capital_scenarios"]]:
        for layer in range(7):
            if layer == 0:
                selected_market = unconstrained_market
                model = zero_model
            elif layer == 1:
                selected_market = ideal_cost_market
                model = ideal_cost_model
            else:
                selected_market = strategy_market
                model = base_cost_model
            result = run_vwap_backtest(
                selected_market,
                layer_targets[layer][["signal_date", "instrument", "target_weight"]],
                capital,
                model,
            )
            metrics, daily = summarize_backtest_performance(
                result.nav,
                benchmark,
                initial_capital=capital,
                annualization_factor=annualization,
            )
            execution = summarize_execution(result.trades, calendar)
            commission = float(result.trades["commission_cost"].sum())
            transfer = float(result.trades["transfer_fee_cost"].sum())
            stamp = float(result.trades["stamp_duty_cost"].sum())
            impact = float(result.trades["impact_cost"].sum())
            summary_rows.append(
                {
                    "capital": capital,
                    "layer": layer,
                    "layer_name": LAYER_NAMES[layer],
                    **metrics,
                    "order_rows": int(execution["order_rows"]),
                    "executed_rows": int(execution["executed_rows"]),
                    "blocked_by_tradability_rows": int(
                        execution["blocked_by_tradability_rows"]
                    ),
                    "participation_limited_rows": int(
                        execution["participation_limited_rows"]
                    ),
                    "cash_limited_rows": int(execution["cash_limited_rows"]),
                    "fill_ratio": float(execution["fill_ratio"]),
                    "t_plus_one_ok": bool(execution["t_plus_one_ok"]),
                    "commission_cost": commission,
                    "transfer_fee_cost": transfer,
                    "stamp_duty_cost": stamp,
                    "impact_cost": impact,
                }
            )
            daily.insert(0, "capital", capital)
            daily.insert(1, "layer", layer)
            trades = result.trades.copy()
            trades.insert(0, "capital", capital)
            trades.insert(1, "layer", layer)
            daily_by_layer[layer].append(daily)
            trades_by_layer[layer].append(trades)
            print(
                f"D8 回测：资金 {capital:,.0f}，第{layer}层，"
                f"年化超额 {metrics['annualized_excess_return']:.2%}",
                flush=True,
            )
    return (
        pd.DataFrame(summary_rows),
        {layer: pd.concat(frames, ignore_index=True) for layer, frames in daily_by_layer.items()},
        {layer: pd.concat(frames, ignore_index=True) for layer, frames in trades_by_layer.items()},
    )


def write_report(
    summary: dict,
    ladder: pd.DataFrame,
    loss_deltas: pd.DataFrame,
    reports_dir: Path,
) -> None:
    main_capital = float(summary["main_capital_cny"])
    main = ladder.loc[np.isclose(ladder["capital"], main_capital)].sort_values("layer")
    target = summary["target_constraint_audit"]
    acceptance = summary["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标"
    lines = [
        "# D8 设计窗口损耗阶梯报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 口径",
        "",
        f"- 主展示资金：{main_capital:,.0f} 元；另同步运行 {summary['capital_scenarios_text']}。",
        "- 信号与组合日期：D5/D7 的 2017–2018 走向前设计期截面；没有读取 2019 年以后数据。",
        "- 第0层为 Top100 等权、零成本、理想成交；第1层只加入真实动态成本；第2层开始启用 D3 的可交易性与5%成交额上限。",
        "- 第3–5层使用与 D7 相同的带线性成本目标，并统一以冻结的 D7 事前持仓为换手参照，逐层只加入行业、Size/Beta、换手约束；第6层直接复用冻结的 D7 目标权重。",
        "- 动态冲击、佣金、过户费、卖出印花税及 T+1 VWAP 全部复用 D3 实现。",
        "- 事前 TE 每个信号日仅使用截至当日的过去252日收益估计 Ledoit–Wolf 协方差，只报告、不调参。",
        "",
        "## D7 前置状态",
        "",
        "- D7 的 98/98 求解、fallback=0 和全部硬约束已通过；D8 第6层与 D7 权重逐行一致。",
        f"- D7 TE 中位数为 {summary['upstream_d7']['tracking_error_median']:.2%}，低于4%下限；本报告保留该失败，不回头修改参数。",
        "",
        "## 七层损耗阶梯（主资金）",
        "",
        "| 层 | 设置 | 年化超额 | 事前TE中位数 | 目标单边换手中位数 | 对应暴露/执行实测 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in main.itertuples(index=False):
        layer = int(row.layer)
        if layer == 0:
            measured = "Top100=100只"
        elif layer == 1:
            measured = f"累计成本/本金={row.transaction_cost_over_initial_capital:.2%}"
        elif layer == 2:
            measured = f"交易受阻={int(row.blocked_by_tradability_rows):,}行"
        elif layer == 3:
            measured = f"行业最大偏离={target[str(layer)]['maximum_industry_active']:.2%}"
        elif layer == 4:
            measured = f"Size/Beta最大偏离={target[str(layer)]['maximum_style_active']:.3f}σ"
        elif layer == 5:
            measured = f"目标换手最大值={target[str(layer)]['maximum_target_turnover']:.2%}"
        else:
            measured = f"个股最大偏离={target[str(layer)]['maximum_stock_active']:.2%}"
        lines.append(
            f"| {layer} | {LAYER_NAMES[layer]} | {row.annualized_excess_return:.2%} "
            f"| {row.ex_ante_te_median:.2%} | {row.target_turnover_median:.2%} | {measured} |"
        )
    lines.extend(
        [
            "",
            "说明：第0–2层目标换手按各层自身路径计算；第3–6层作为边际约束归因，统一相对冻结的 D7 事前持仓计算，避免放松个股约束后形成的集中路径令后续15%换手集合为空。实际成交换手另行记录，会受停牌、涨跌停、容量和现金约束影响。",
            "",
            "## 相邻层变化与解释",
            "",
            "| 新层 | 年化超额较上一层变化 | 固定解释 |",
            "|---:|---:|---|",
        ]
    )
    for row in loss_deltas.itertuples(index=False):
        if int(row.layer) == 0:
            continue
        lines.append(
            f"| {int(row.layer)} | {row.change_vs_previous:+.2%} | {row.explanation} |"
        )
    lines.extend(
        [
            "",
            "损耗不要求单调：行业或风格中性化可能剔除亏损的非预期暴露，从而提高收益；判断重点是约束是否真实生效。",
            "",
            "## 约束与执行验收",
            "",
            f"- 第1层真实成本非零且相对第0层产生净损耗：{'通过' if acceptance['cost_layer_effective'] else '未通过'}。",
            f"- 第0/1层未混入停牌或容量限制：{'通过' if acceptance['cost_layer_isolated'] else '未通过'}。",
            f"- 第2层启用真实可交易性并产生受阻记录：{'通过' if acceptance['tradability_layer_effective'] else '未通过'}。",
            f"- 第3–5层求解100%成功且无 fallback：{'通过' if acceptance['layer_3_5_solver_success_100pct'] and acceptance['layer_3_5_fallback_zero'] else '未通过'}。",
            f"- 第3层起行业偏离≤2%：{'通过' if acceptance['industry_constraint'] else '未通过'}。",
            f"- 第4层起 Size/Beta 偏离≤0.3σ：{'通过' if acceptance['style_constraint'] else '未通过'}。",
            f"- 第5层起目标单边换手≤15%：{'通过' if acceptance['turnover_constraint'] else '未通过'}。",
            f"- 第6层个股偏离≤1.5%：{'通过' if acceptance['stock_constraint'] else '未通过'}。",
            f"- 第5层目标换手不高于第4层：{'通过' if acceptance['turnover_changed_as_expected'] else '未通过'}。",
            f"- 所有层均为 T+1 撮合：{'通过' if acceptance['t_plus_one_all_layers'] else '未通过'}。",
            "",
            "## 输出",
            "",
            "- `reports/d8_ladder_summary.parquet`、`reports/d8_loss_deltas.parquet`",
            "- `reports/d8_target_audit.parquet`、`reports/d8_factor_active_exposures.parquet`",
            "- `reports/d8_ex_ante_te.parquet`、`reports/d8_solver_audit.parquet`",
            "- 每层分别保存 `d8_layer_N_targets.parquet`、`d8_layer_N_daily.parquet`、`d8_layer_N_trades.parquet`",
            "- `reports/d8_summary.json`",
            "",
            "## 下一步",
            "",
            "冻结 D1–D8 代码和参数后才能进入 D9。D7 的 TE 前置门槛未通过，必须作为 D9 结论中的风险提示保留。",
            "",
        ]
    )
    path = reports_dir / "D8设计窗口损耗阶梯报告.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = load_config()
    layer_ids, main_capital = validate_config(config)
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D8 必须在 design 模式运行，不得解封 OOS 数据")
    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    reports_dir = ROOT / config["data"]["reports_dir"]
    raw_dir = Path(config["data"]["tushare_raw_dir"])
    processed_dir = Path(config["data"]["processed_design_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    d3_summary = load_json(
        reports_dir / "d3_summary.json", "缺少 d3_summary.json，请先成功运行 D3"
    )
    d6_summary = load_json(
        reports_dir / "d6_summary.json", "缺少 d6_summary.json，请先成功运行 D6"
    )
    d7_summary = load_json(
        reports_dir / "d7_summary.json", "缺少 d7_summary.json，请先完成 D7"
    )
    if not d3_summary.get("acceptance", {}).get("passed", False):
        raise RuntimeError("D3 未通过，不得进入 D8")
    if not d6_summary.get("acceptance", {}).get("passed", False):
        raise RuntimeError("D6 未通过，不得进入 D8")
    if d3_summary.get("mode") != "design":
        raise RuntimeError("D3 不是设计窗口产物，不得进入 D8")
    if (
        d6_summary.get("mode") != "design"
        or d6_summary.get("oos_data_accessed") is not False
    ):
        raise RuntimeError("D6 不是封存设计窗口产物，不得进入 D8")
    d7_status = require_d7_structural_completion(d7_summary)

    predictions = load_design_parquet(
        access,
        Path(config["data"]["reports_dir"]) / "d5_walk_forward_predictions.parquet",
        date_col="sample_date",
        start_date=design_start,
        end_date=design_end,
    )
    exposures = load_design_parquet(
        access,
        Path(config["data"]["reports_dir"]) / "d6_factor_exposures.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    d7_targets = load_design_parquet(
        access,
        Path(config["data"]["reports_dir"]) / "d7_target_weights.parquet",
        date_col="signal_date",
        start_date=design_start,
        end_date=design_end,
    )
    weights = load_design_parquet(
        access,
        processed_dir / "benchmark_weights.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    tradability = load_design_parquet(
        access,
        processed_dir / "tradability_flags.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    if predictions.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("D5 预测存在重复日期-股票")
    if exposures.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("D6 暴露存在重复日期-股票")
    signal_dates = pd.DatetimeIndex(predictions["sample_date"].unique()).sort_values()
    if len(signal_dates) == 0:
        raise ValueError("D5 走向前预测没有设计窗口信号日")
    if not signal_dates.equals(
        pd.DatetimeIndex(d7_targets["signal_date"].unique()).sort_values()
    ):
        raise ValueError("D5 与 D7 信号日期不一致")
    industry_columns = [str(value) for value in d6_summary["factors"]["industry_columns"]]
    style_columns = [
        f"style_{name}" for name in config["d7"]["constrained_style_factors"]
    ]

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
    instruments = sorted(weights["instrument"].astype(str).unique())
    print("读取 D8 全套 D3 行情与可交易性数据……", flush=True)
    market = load_qlib_market(
        access,
        instruments,
        design_start,
        design_end,
        float(config["tradability"]["qlib_amount_multiplier_to_cny"]),
    )
    limits = load_yearly(access, config, "stk_limit", design_start.year, design_end.year)
    suspensions = load_yearly(
        access, config, "suspend_d", design_start.year, design_end.year
    )
    market = add_exchange_flags(market, limits, suspensions, config)
    market = add_ex_ante_volatility(market, config)
    strategy_market, _ = prepare_markets(market, tradability)
    fallback_amount = float(config["tradability"]["min_avg_daily_amount_cny"])
    ideal_cost_market = make_ideal_execution_market(
        strategy_market, fallback_daily_amount=fallback_amount
    )
    # 第0与第1层使用完全相同的理想成交行情，确保两层差异只来自成本。
    # 5%容量约束从第2层与真实可交易性一起启用。
    unconstrained_market = ideal_cost_market.copy()

    execution_schedule = make_execution_schedule(signal_dates, calendar)
    execution_by_signal = execution_schedule.set_index("signal_date")["execution_date"]
    d7_schedule = (
        d7_targets[["signal_date", "execution_date"]]
        .drop_duplicates()
        .sort_values("signal_date")
        .reset_index(drop=True)
    )
    if not d7_schedule.equals(
        execution_schedule.sort_values("signal_date").reset_index(drop=True)
    ):
        raise ValueError("D7 目标中的 T+1 执行日期与 D8 交易日历不一致")
    price_matrix = build_execution_price_matrix(market)
    frozen_pretrade_by_signal = build_frozen_d7_pretrade_path(
        d7_targets,
        exposures,
        signal_dates,
        execution_by_signal,
        price_matrix,
        tolerance=float(config["d7"]["constraint_tolerance"]),
    )
    top_targets = make_top_n_equal_targets(
        predictions,
        top_n=int(config["d8"]["top_n"]),
        score_col=str(config["d5"]["expected_return_column"]),
    )
    layer_targets: dict[int, pd.DataFrame] = {
        0: top_targets.copy(),
        1: top_targets.copy(),
        2: top_targets.copy(),
    }
    solver_frames: list[pd.DataFrame] = []
    for layer in [3, 4, 5]:
        targets, solver_audit = generate_optimized_layer_targets(
            layer,
            predictions,
            exposures,
            signal_dates,
            execution_by_signal,
            frozen_pretrade_by_signal,
            industry_columns,
            config,
        )
        layer_targets[layer] = targets
        solver_frames.append(solver_audit)
    layer_targets[6] = d7_targets[
        ["signal_date", "execution_date", "instrument", "target_weight"]
    ].copy()
    layer6_match = assert_layer6_matches_d7(
        layer_targets[6],
        d7_targets,
        tolerance=float(config["d8"]["layer6_target_match_tolerance"]),
    )

    target_audit, factor_active = audit_all_layer_targets(
        layer_targets,
        exposures,
        signal_dates,
        execution_by_signal,
        price_matrix,
        frozen_pretrade_by_signal,
        industry_columns,
        style_columns,
        float(config["d7"]["constraint_tolerance"]),
    )
    stock_returns = build_stock_returns(market)
    ex_ante_te = calculate_layer_tracking_error(
        layer_targets,
        exposures,
        stock_returns,
        signal_dates,
        config,
    )
    benchmark = load_design_parquet(
        access,
        raw_dir / "index_daily_H00905_CSI.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    ladder, daily_by_layer, trades_by_layer = run_loss_ladder_backtests(
        layer_targets,
        unconstrained_market,
        ideal_cost_market,
        strategy_market,
        benchmark,
        calendar,
        config,
    )

    target_summary = (
        target_audit.groupby("layer")
        .agg(
            maximum_stock_active=("maximum_stock_active", "max"),
            maximum_industry_active=("maximum_industry_active", "max"),
            maximum_style_active=("maximum_style_active", "max"),
            maximum_target_turnover=("target_single_sided_turnover", "max"),
            target_turnover_median=("target_single_sided_turnover", "median"),
            minimum_holdings=("holdings", "min"),
            median_holdings=("holdings", "median"),
            maximum_holdings=("holdings", "max"),
        )
        .reset_index()
    )
    te_summary = (
        ex_ante_te.groupby("layer")
        .agg(
            ex_ante_te_median=("annualized_ex_ante_tracking_error", "median"),
            ex_ante_te_mean=("annualized_ex_ante_tracking_error", "mean"),
            ex_ante_te_minimum=("annualized_ex_ante_tracking_error", "min"),
            ex_ante_te_maximum=("annualized_ex_ante_tracking_error", "max"),
            covariance_active_l1_coverage_min=("covariance_active_l1_coverage", "min"),
        )
        .reset_index()
    )
    ladder = ladder.merge(target_summary, on="layer", how="left", validate="many_to_one")
    ladder = ladder.merge(te_summary, on="layer", how="left", validate="many_to_one")
    loss_deltas = build_loss_deltas(ladder, capital=main_capital)
    main = ladder.loc[np.isclose(ladder["capital"], main_capital)].set_index("layer")
    solver_audit = pd.concat(solver_frames, ignore_index=True)
    tolerance = float(config["d7"]["constraint_tolerance"])
    acceptance = {
        "seven_layers_complete": bool(sorted(layer_targets) == list(layer_ids)),
        "layer6_matches_frozen_d7": bool(layer6_match["passed"]),
        "cost_layer_effective": bool(
            main.loc[0, "total_transaction_cost"] == 0
            and main.loc[1, "total_transaction_cost"] > 0
            and main.loc[1, "annualized_excess_return"]
            < main.loc[0, "annualized_excess_return"]
        ),
        "cost_layer_isolated": bool(
            ladder.loc[ladder["layer"].le(1), "blocked_by_tradability_rows"]
            .eq(0)
            .all()
            and ladder.loc[
                ladder["layer"].le(1), "participation_limited_rows"
            ]
            .eq(0)
            .all()
        ),
        "dynamic_impact_used": bool(
            ladder.loc[ladder["layer"].ge(1), "impact_cost"].gt(0).all()
        ),
        "tradability_layer_effective": bool(
            main.loc[2, "blocked_by_tradability_rows"] > 0
            and main.loc[2, "blocked_by_tradability_rows"]
            > main.loc[1, "blocked_by_tradability_rows"]
        ),
        "layer_3_5_solver_success_100pct": bool(
            solver_audit["solver_status"].eq("optimal").all()
        ),
        "layer_3_5_fallback_zero": bool(
            ~solver_audit["fallback_used"].any()
        ),
        "industry_constraint": bool(
            target_summary.loc[
                target_summary["layer"].ge(3), "maximum_industry_active"
            ].max()
            <= float(config["d7"]["industry_active_limit"]) + tolerance
        ),
        "style_constraint": bool(
            target_summary.loc[
                target_summary["layer"].ge(4), "maximum_style_active"
            ].max()
            <= float(config["d7"]["style_active_limit"]) + tolerance
        ),
        "turnover_constraint": bool(
            target_summary.loc[
                target_summary["layer"].ge(5), "maximum_target_turnover"
            ].max()
            <= float(config["d7"]["single_sided_turnover_limit"]) + tolerance
        ),
        "stock_constraint": bool(
            float(target_summary.loc[target_summary["layer"].eq(6), "maximum_stock_active"].iloc[0])
            <= float(config["d7"]["stock_active_limit"]) + tolerance
        ),
        "turnover_changed_as_expected": bool(
            main.loc[5, "target_turnover_median"]
            <= main.loc[4, "target_turnover_median"] + tolerance
        ),
        "t_plus_one_all_layers": bool(ladder["t_plus_one_ok"].all()),
        "explanations_complete": bool(
            loss_deltas.loc[loss_deltas["layer"].ge(1), "explanation"].ne("").all()
        ),
        "oos_sealed": True,
    }
    acceptance["passed"] = bool(all(acceptance.values()))

    capital_scenarios = [float(value) for value in config["portfolio"]["capital_scenarios"]]
    target_json = {
        str(int(row.layer)): {
            "maximum_stock_active": float(row.maximum_stock_active),
            "maximum_industry_active": float(row.maximum_industry_active),
            "maximum_style_active": float(row.maximum_style_active),
            "maximum_target_turnover": float(row.maximum_target_turnover),
            "target_turnover_median": float(row.target_turnover_median),
            "minimum_holdings": int(row.minimum_holdings),
            "median_holdings": float(row.median_holdings),
            "maximum_holdings": int(row.maximum_holdings),
        }
        for row in target_summary.itertuples(index=False)
    }
    main_rows = ladder.loc[np.isclose(ladder["capital"], main_capital)].sort_values("layer")
    summary = {
        "mode": access.mode,
        "oos_data_accessed": False,
        "main_capital_cny": main_capital,
        "optimized_turnover_reference": "frozen_d7_pretrade_path",
        "capital_scenarios": capital_scenarios,
        "capital_scenarios_text": "、".join(f"{value:,.0f}元" for value in capital_scenarios),
        "signal_window": {
            "start": str(signal_dates.min().date()),
            "end": str(signal_dates.max().date()),
            "dates": int(len(signal_dates)),
        },
        "upstream_d3_passed": True,
        "upstream_d6_passed": True,
        "upstream_d7": {
            **d7_status,
            "tracking_error_median": float(d7_summary["tracking_error"]["median"]),
            "tracking_error_acceptance_min": float(
                d7_summary["tracking_error"]["acceptance_min"]
            ),
        },
        "layer6_match": layer6_match,
        "layers": [
            {
                "layer": int(row.layer),
                "name": row.layer_name,
                "annualized_excess_return": float(row.annualized_excess_return),
                "annualized_portfolio_return": float(row.portfolio_annualized_return),
                "annualized_benchmark_return": float(row.benchmark_annualized_return),
                "ex_ante_te_median": float(row.ex_ante_te_median),
                "target_turnover_median": float(row.target_turnover_median),
                "executed_turnover_mean": float(row.mean_rebalance_single_sided_turnover),
                "transaction_cost": float(row.total_transaction_cost),
                "impact_cost": float(row.impact_cost),
                "blocked_by_tradability_rows": int(row.blocked_by_tradability_rows),
                "fill_ratio": float(row.fill_ratio),
                "explanation": LAYER_EXPLANATIONS[int(row.layer)],
            }
            for row in main_rows.itertuples(index=False)
        ],
        "target_constraint_audit": target_json,
        "acceptance": acceptance,
    }

    atomic_parquet(ladder, reports_dir / "d8_ladder_summary.parquet")
    atomic_parquet(loss_deltas, reports_dir / "d8_loss_deltas.parquet")
    atomic_parquet(target_audit, reports_dir / "d8_target_audit.parquet")
    atomic_parquet(
        factor_active, reports_dir / "d8_factor_active_exposures.parquet"
    )
    atomic_parquet(ex_ante_te, reports_dir / "d8_ex_ante_te.parquet")
    atomic_parquet(solver_audit, reports_dir / "d8_solver_audit.parquet")
    for layer in range(7):
        atomic_parquet(
            layer_targets[layer], reports_dir / f"d8_layer_{layer}_targets.parquet"
        )
        atomic_parquet(
            daily_by_layer[layer], reports_dir / f"d8_layer_{layer}_daily.parquet"
        )
        atomic_parquet(
            trades_by_layer[layer], reports_dir / f"d8_layer_{layer}_trades.parquet"
        )
    atomic_json(summary, reports_dir / "d8_summary.json")
    write_report(summary, ladder, loss_deltas, reports_dir)

    print(f"D8 验收：{'达标' if acceptance['passed'] else '未达标'}")
    for row in main_rows.itertuples(index=False):
        print(
            f"第{int(row.layer)}层：年化超额 {row.annualized_excess_return:.2%}，"
            f"事前TE {row.ex_ante_te_median:.2%}，目标换手 {row.target_turnover_median:.2%}"
        )
    print(f"报告：{reports_dir / 'D8设计窗口损耗阶梯报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

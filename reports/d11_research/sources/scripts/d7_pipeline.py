from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.d3_core import CostModel  # noqa: E402
from src.d6_core import estimate_ledoit_wolf_covariance  # noqa: E402
from src.d7_core import (  # noqa: E402
    OptimizationLimits,
    annualized_tracking_error,
    drift_pretrade_weights,
    optimize_long_only_active_portfolio,
    symmetric_l1_cost_rate,
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def require_passed_d6(reports_dir: Path) -> dict:
    path = reports_dir / "d6_summary.json"
    if not path.exists():
        raise FileNotFoundError("缺少 reports/d6_summary.json，请先成功运行 D6")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "design":
        raise RuntimeError("D6 结果不是 design 模式产物")
    if not payload.get("acceptance", {}).get("passed", False):
        raise RuntimeError("D6 未通过写死的验收标准，不得进入 D7")
    return payload


def validate_d7_config(config: dict) -> tuple[OptimizationLimits, list[str]]:
    d7 = config["d7"]
    expected = {
        "solver": "CLARABEL",
        "initial_portfolio": "benchmark",
        "objective_cost_method": "symmetric_linear_fees_and_half_stamp",
        "stock_active_limit": 0.015,
        "industry_active_limit": 0.02,
        "style_active_limit": 0.30,
        "single_sided_turnover_limit": 0.15,
    }
    for key, value in expected.items():
        actual = d7.get(key)
        if isinstance(value, float):
            valid = np.isclose(float(actual), value, rtol=0.0, atol=1e-15)
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"D7 必须遵守整体计划：d7.{key} 应为 {value!r}")
    styles = [str(item) for item in d7["constrained_style_factors"]]
    if styles != ["Size", "Beta"]:
        raise ValueError("D7 风格硬约束必须只卡死 Size 与 Beta")
    if d7["tracking_error_acceptance_statistic"] != "median":
        raise ValueError("D7 事前 TE 验收统计量必须固定为周频中位数")
    limits = OptimizationLimits(
        stock_active=float(d7["stock_active_limit"]),
        industry_active=float(d7["industry_active_limit"]),
        style_active=float(d7["style_active_limit"]),
        single_sided_turnover=float(d7["single_sided_turnover_limit"]),
        tolerance=float(d7["constraint_tolerance"]),
    )
    limits.validate()
    return limits, [f"style_{name}" for name in styles]


def load_market_prices(
    access: ResearchDataAccess,
    instruments: list[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    raw = access.features(instruments, ["$vwap", "$close"], start_date, end_date)
    if raw.empty:
        raise ValueError("Qlib 没有返回 D7 所需 VWAP/收盘价")
    raw = raw.copy()
    raw.columns = ["vwap", "close"]
    frame = raw.reset_index().rename(columns={"datetime": "trade_date"})
    frame["trade_date"] = parse_date_series(frame["trade_date"])
    for column in ["vwap", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~frame[column].gt(0), column] = np.nan
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("D7 行情存在重复的日期-股票记录")
    return frame.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def make_execution_schedule(
    signal_dates: pd.DatetimeIndex, trading_calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(trading_calendar).normalize().sort_values().unique()
    rows = []
    for signal_date in pd.DatetimeIndex(signal_dates).normalize().sort_values().unique():
        later = calendar[calendar > signal_date]
        if later.empty:
            raise ValueError(f"{signal_date.date()} 之后没有 T+1 执行日")
        rows.append({"signal_date": signal_date, "execution_date": later[0]})
    return pd.DataFrame(rows)


def build_execution_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Build causal execution/valuation marks for portfolio-weight drift.

    A traded stock uses that day's VWAP.  If VWAP is unavailable but a close is
    published, the same-day close is used for valuation only.  If a suspension
    leaves both fields empty, carry forward the latest known close.  Never
    backward-fill, so a later resumption price cannot leak into an earlier day.
    """

    required = {"trade_date", "instrument", "vwap", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"执行估值价格缺少字段：{sorted(missing)}")
    frame = prices[list(required)].copy()
    frame["trade_date"] = parse_date_series(frame["trade_date"])
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("执行估值价格存在重复日期-股票")
    vwap = frame.pivot(index="trade_date", columns="instrument", values="vwap")
    close = frame.pivot(index="trade_date", columns="instrument", values="close")
    vwap = vwap.sort_index()
    close = close.reindex(index=vwap.index, columns=vwap.columns)
    same_day = vwap.combine_first(close)
    latest_known_close = close.ffill()
    return same_day.combine_first(latest_known_close)


def holding_returns_between(
    price_matrix: pd.DataFrame,
    start_date,
    end_date,
    held_instruments: pd.Index,
) -> pd.Series:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start not in price_matrix.index or end not in price_matrix.index:
        raise ValueError(f"持仓漂移缺少执行日价格截面：{start.date()} 或 {end.date()}")
    start_price = price_matrix.loc[start].reindex(held_instruments)
    end_price = price_matrix.loc[end].reindex(held_instruments)
    invalid = (
        start_price.isna()
        | end_price.isna()
        | ~start_price.gt(0)
        | ~end_price.gt(0)
    )
    if invalid.any():
        missing = held_instruments[invalid].tolist()
        raise ValueError(
            f"{start.date()}→{end.date()} 持仓漂移缺少有效价格："
            + ", ".join(map(str, missing[:20]))
        )
    result = end_price / start_price - 1.0
    result.name = "holding_period_return"
    return result


def build_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    result = (
        prices.pivot(index="trade_date", columns="instrument", values="close")
        .pct_change(fill_method=None)
        .stack(future_stack=True)
        .rename("return")
        .reset_index()
    )
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def select_signal_dates(
    predictions: pd.DataFrame, exposures: pd.DataFrame
) -> pd.DatetimeIndex:
    predicted = pd.DatetimeIndex(predictions["sample_date"].unique()).normalize()
    exposed = pd.DatetimeIndex(exposures["trade_date"].unique()).normalize()
    missing = predicted.difference(exposed)
    if len(missing):
        raise ValueError(
            "D5 预测日缺少 D6 暴露："
            + ", ".join(str(item.date()) for item in missing[:20])
        )
    return predicted.sort_values().unique()


def write_report(summary: dict, reports_dir: Path) -> None:
    acceptance = summary["acceptance"]
    constraints = summary["constraint_audit"]
    te = summary["tracking_error"]
    coverage = summary["signal_coverage"]
    status = "达标" if acceptance["passed"] else "未达标"
    lines = [
        "# D7 组合优化器报告",
        "",
        f"**验收结论：{status}**",
        "",
        "## 固定口径",
        "",
        "- 求解器：cvxpy + Clarabel；每个截面只求解一次，不设置任何放松约束的 fallback。",
        "- 目标：最大化 `dᵀμ − c‖w−w₀‖₁`；μ 来自 D5 校准后的未来 5 日预期超额。",
        "- `c = 佣金 + 过户费 + 0.5×卖出印花税`。在全投资且无外部现金流时，买入额等于卖出额，因此该对称系数与实际线性费用严格等价。",
        "- 动态冲击成本不是线性的，不伪装成 LP 项；它按总计划留在 D8 的真实成本层由 D3 成本模型逐笔计算。",
        "- 首个截面以当日基准权重作为初始组合；以后将上一期目标权重从上期 T+1 漂移到本期 T+1：有成交用当日 VWAP，无 VWAP 用当日收盘价，停牌导致两者均缺失时使用截至当日最近可得收盘价。全程只向前填充，不使用未来复牌价。",
        "- 行业、Size、Beta 风险使用线性硬约束；Ledoit–Wolf 协方差只报告事前 TE，不进入目标或约束。",
        "",
        "## 日期覆盖说明",
        "",
        f"- D6 暴露覆盖 {coverage['d6_exposure_start']} 至 {coverage['d6_exposure_end']}。",
        f"- D5 走向前 μ 覆盖 {coverage['optimization_start']} 至 {coverage['optimization_end']}，共 {coverage['optimization_dates']} 个周频截面；D7 已全部优化。",
        "- D5 按计划只用 2014–2016 训练、2017–2018 走向前预测，因此 2014–2016 没有可用于组合优化的 μ。这里把“2014–2018 全区间”严格解释为设计窗口内全部可用走向前信号日，不伪造前三年的预测。",
        "",
        "## 求解与实际约束审计",
        "",
        f"- 严格 optimal：{summary['solver']['optimal_dates']}/{coverage['optimization_dates']}；成功率 {summary['solver']['success_rate']:.2%}。",
        f"- fallback 触发：{summary['solver']['fallback_count']} 次。",
        "",
        "| 约束 | 计划上限 | 全期实测最大值 | 通过 |",
        "|---|---:|---:|:---:|",
        f"| 权重和误差 | {summary['limits']['constraint_tolerance']:.1e} | {constraints['maximum_weight_sum_error']:.3e} | {'是' if acceptance['fully_invested'] else '否'} |",
        f"| 最小权重 | ≥ 0 | {constraints['minimum_weight']:.3e} | {'是' if acceptance['long_only'] else '否'} |",
        f"| 个股主动偏离 | {summary['limits']['stock_active']:.2%} | {constraints['maximum_stock_active']:.2%} | {'是' if acceptance['stock_active'] else '否'} |",
        f"| 行业主动偏离 | {summary['limits']['industry_active']:.2%} | {constraints['maximum_industry_active']:.2%} | {'是' if acceptance['industry_active'] else '否'} |",
        f"| Size / Beta 主动暴露 | {summary['limits']['style_active']:.2f}σ | {constraints['maximum_style_active']:.3f}σ | {'是' if acceptance['style_active'] else '否'} |",
        f"| 单边换手 | {summary['limits']['single_sided_turnover']:.2%} | {constraints['maximum_single_sided_turnover']:.2%} | {'是' if acceptance['turnover'] else '否'} |",
        "",
        f"- μ 缺失股票按 0（无主动观点）处理；最差截面信号覆盖率 {coverage['minimum_mu_coverage']:.2%}，不会把缺失误当负分。",
        f"- 目标持仓数中位数 {summary['portfolio']['median_holdings']:.0f}，范围 {summary['portfolio']['minimum_holdings']}–{summary['portfolio']['maximum_holdings']}（本层不设置非凸持仓数约束）。",
        "",
        "## 报告用事前跟踪误差",
        "",
        f"- 周频年化 TE 中位数 {te['median']:.2%}，均值 {te['mean']:.2%}，范围 {te['minimum']:.2%}–{te['maximum']:.2%}。",
        f"- 验收区间为 {te['acceptance_min']:.2%}–{te['acceptance_max']:.2%}，按配置固定使用中位数判定：{'通过' if acceptance['tracking_error'] else '未通过'}。",
        f"- 协方差对主动权重 L1 的最低覆盖率 {te['minimum_active_l1_coverage']:.2%}：{'通过' if acceptance['covariance_coverage'] else '未通过'}。",
        "- 该 Σ 未做 Bias Test 校准，TE 只用于本轮诊断，不作实盘风险承诺。",
        "",
        "## 输出",
        "",
        "- `reports/d7_target_weights.parquet`",
        "- `reports/d7_constraint_audit.parquet`",
        "- `reports/d7_factor_active_exposures.parquet`",
        "- `reports/d7_covariance_audit.parquet`",
        "- `reports/d7_summary.json`",
        "",
        "## 下一步",
        "",
        "D7 通过后进入 D8：在设计窗口内搭建七层损耗阶梯，并用 D3 撮合与动态成本模型验证优化目标落到实际成交后的损耗。",
        "",
    ]
    path = reports_dir / "D7组合优化器报告.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = load_config()
    d6 = config["d6"]
    d7 = config["d7"]
    limits, constrained_style_columns = validate_d7_config(config)
    reports_dir = ROOT / config["data"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    access = ResearchDataAccess(ROOT)
    if access.mode != "design":
        raise RuntimeError("D7 必须在 design 模式运行，不得解封 OOS 数据")
    d6_summary = require_passed_d6(reports_dir)
    industry_columns = [str(item) for item in d6_summary["factors"]["industry_columns"]]
    if any(column not in d6_summary["factors"]["style_columns"] for column in constrained_style_columns):
        raise ValueError("D6 输出缺少 D7 要约束的 Size/Beta 风格列")

    design_start = pd.Timestamp(config["windows"]["design_start"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    predictions = access.read_parquet(
        Path(config["data"]["reports_dir"]) / "d5_walk_forward_predictions.parquet",
        date_col="sample_date",
        start_date=design_start,
        end_date=design_end,
    )
    exposures = access.read_parquet(
        Path(config["data"]["reports_dir"]) / "d6_factor_exposures.parquet",
        date_col="trade_date",
        start_date=design_start,
        end_date=design_end,
    )
    predictions["sample_date"] = parse_date_series(predictions["sample_date"])
    exposures["trade_date"] = parse_date_series(exposures["trade_date"])
    if predictions.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("D5 预测存在重复日期-股票")
    if exposures.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("D6 暴露存在重复日期-股票")
    expected_return_column = str(config["d5"]["expected_return_column"])
    if expected_return_column not in predictions:
        raise ValueError(f"D5 预测缺少 {expected_return_column}")
    signal_dates = select_signal_dates(predictions, exposures)
    if signal_dates.empty:
        raise ValueError("设计窗口内没有 D5/D6 共同周频截面")

    instruments = sorted(exposures["instrument"].astype(str).unique())
    history_start = signal_dates.min() - pd.Timedelta(
        days=int(d6["history_buffer_calendar_days"])
    )
    print("读取 D7 持仓漂移与逐期事前 TE 所需行情……", flush=True)
    prices = load_market_prices(access, instruments, history_start, design_end)
    trading_calendar = pd.DatetimeIndex(prices["trade_date"].unique())
    execution_schedule = make_execution_schedule(signal_dates, trading_calendar)
    execution_by_signal = execution_schedule.set_index("signal_date")["execution_date"]
    price_matrix = build_execution_price_matrix(prices)
    stock_returns = build_stock_returns(prices)

    cost_model = CostModel.from_config(config)
    target_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    covariance_rows: list[dict[str, object]] = []
    previous_target: pd.Series | None = None
    previous_execution_date: pd.Timestamp | None = None
    fallback_count = 0

    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross_section = exposures.loc[exposures["trade_date"].eq(signal_date)].copy()
        cross_section = cross_section.set_index("instrument").sort_index()
        benchmark = pd.to_numeric(
            cross_section["benchmark_weight"], errors="coerce"
        ).astype(float)
        if benchmark.isna().any() or (benchmark < 0).any():
            raise ValueError(f"{signal_date.date()} 基准权重无效")
        benchmark /= benchmark.sum()
        prediction = predictions.loc[predictions["sample_date"].eq(signal_date)].copy()
        unexpected = sorted(set(prediction["instrument"]) - set(cross_section.index))
        if unexpected:
            raise ValueError(
                f"{signal_date.date()} D5 含非当日成分：" + ", ".join(unexpected[:20])
            )
        mu_observed = prediction.set_index("instrument")[expected_return_column]
        mu = pd.to_numeric(mu_observed, errors="coerce").reindex(cross_section.index)
        invalid_observed = mu.loc[mu.index.isin(mu_observed.index)].isna()
        if invalid_observed.any():
            raise ValueError(f"{signal_date.date()} D5 已输出的 μ 含非有限值")
        mu_coverage = float(mu.notna().mean())
        mu = mu.fillna(0.0).astype(float)

        if previous_target is None:
            pretrade = benchmark.copy()
            pretrade_source = "benchmark_seed"
        else:
            held = previous_target[previous_target > limits.tolerance * 0.1].index
            holding_returns = holding_returns_between(
                price_matrix,
                previous_execution_date,
                execution_date,
                held,
            )
            pretrade = drift_pretrade_weights(
                previous_target, holding_returns, weight_tolerance=1e-8
            )
            pretrade_source = "prior_target_drifted_execution_to_execution"

        cost_rate = symmetric_l1_cost_rate(
            cost_model.commission_rate,
            cost_model.transfer_fee_rate,
            cost_model.stamp_rate(execution_date),
        )
        result = optimize_long_only_active_portfolio(
            mu,
            benchmark,
            pretrade,
            cross_section,
            industry_columns,
            constrained_style_columns,
            cost_rate=cost_rate,
            limits=limits,
            solver=str(d7["solver"]),
        )

        active = result.weights - benchmark
        covariance = estimate_ledoit_wolf_covariance(
            stock_returns,
            as_of_date=signal_date,
            instruments=cross_section.index,
            window=int(d6["covariance_window_trade_days"]),
            minimum_observations=int(d6["covariance_minimum_observations"]),
        )
        te = annualized_tracking_error(
            active,
            covariance.covariance,
            covariance.instruments,
            annualization_factor=float(d7["covariance_annualization_factor"]),
        )

        target = pd.DataFrame(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "instrument": cross_section.index,
                "target_weight": result.weights,
                "benchmark_weight": benchmark,
                "active_weight": active,
                "pretrade_weight": pretrade.reindex(cross_section.index, fill_value=0.0),
                expected_return_column: mu,
                "mu_observed": mu.index.isin(mu_observed.index),
            }
        ).reset_index(drop=True)
        target_rows.append(target)
        audit = result.audit
        audit_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "solver_status": result.status,
                "fallback_used": False,
                "pretrade_source": pretrade_source,
                "cost_rate_per_l1": cost_rate,
                "mu_coverage": mu_coverage,
                "objective_value": result.objective_value,
                "expected_5d_active_return": result.expected_active_return,
                "estimated_linear_cost": result.estimated_linear_cost,
                "weight_sum_error": audit["weight_sum_error"],
                "minimum_weight": audit["minimum_weight"],
                "maximum_stock_active": audit["maximum_stock_active"],
                "maximum_industry_active": audit["maximum_industry_active"],
                "maximum_style_active": audit["maximum_style_active"],
                "total_l1_trade": audit["total_l1_trade"],
                "single_sided_turnover": audit["single_sided_turnover"],
                "forced_exit_weight": audit["forced_exit_weight"],
                "holdings": int((result.weights > limits.tolerance).sum()),
                "fully_invested_passed": audit["checks"]["fully_invested"],
                "long_only_passed": audit["checks"]["long_only"],
                "stock_active_passed": audit["checks"]["stock_active"],
                "industry_active_passed": audit["checks"]["industry_active"],
                "style_active_passed": audit["checks"]["style_active"],
                "turnover_passed": audit["checks"]["single_sided_turnover"],
                "all_constraints_passed": audit["passed"],
                "annualized_tracking_error": te.annualized_tracking_error,
                "covariance_active_l1_coverage": te.active_l1_coverage,
            }
        )
        for factor_type, values in [
            ("industry", audit["industry_active_exposures"]),
            ("style", audit["style_active_exposures"]),
        ]:
            factor_rows.extend(
                {
                    "signal_date": signal_date,
                    "factor": factor,
                    "factor_type": factor_type,
                    "active_exposure": float(value),
                }
                for factor, value in values.items()
            )
        covariance_rows.append(
            {
                "signal_date": signal_date,
                "observations": covariance.observations,
                "included_instruments": len(covariance.instruments),
                "excluded_instruments": len(covariance.excluded_instruments),
                "shrinkage": covariance.shrinkage,
                "annualized_tracking_error": te.annualized_tracking_error,
                "active_l1_coverage": te.active_l1_coverage,
            }
        )
        previous_target = result.weights
        previous_execution_date = execution_date
        print(
            f"D7 优化 {number}/{len(signal_dates)}：{signal_date.date()}，"
            f"换手 {audit['single_sided_turnover']:.2%}，TE {te.annualized_tracking_error:.2%}",
            flush=True,
        )

    targets = pd.concat(target_rows, ignore_index=True)
    audits = pd.DataFrame(audit_rows).sort_values("signal_date")
    factor_exposures = pd.DataFrame(factor_rows).sort_values(["signal_date", "factor"])
    covariance_audit = pd.DataFrame(covariance_rows).sort_values("signal_date")
    te_statistic = float(audits["annualized_tracking_error"].median())
    te_min = float(d7["tracking_error_acceptance_min"])
    te_max = float(d7["tracking_error_acceptance_max"])
    covariance_coverage_min = float(d7["covariance_minimum_active_l1_coverage"])
    acceptance = {
        "upstream_d6": True,
        "all_available_design_signal_dates": bool(len(audits) == len(signal_dates)),
        "solver_success_100pct": bool(audits["solver_status"].eq("optimal").all()),
        "fallback_zero": bool(fallback_count == 0 and ~audits["fallback_used"].any()),
        "fully_invested": bool(audits["fully_invested_passed"].all()),
        "long_only": bool(audits["long_only_passed"].all()),
        "stock_active": bool(audits["stock_active_passed"].all()),
        "industry_active": bool(audits["industry_active_passed"].all()),
        "style_active": bool(audits["style_active_passed"].all()),
        "turnover": bool(audits["turnover_passed"].all()),
        "covariance_coverage": bool(
            audits["covariance_active_l1_coverage"].min() >= covariance_coverage_min
        ),
        "tracking_error": bool(te_min <= te_statistic <= te_max),
    }
    acceptance["passed"] = bool(all(acceptance.values()))

    try:
        import clarabel
        import cvxpy

        versions = {"cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__}
    except (ImportError, AttributeError):  # pragma: no cover
        versions = {"cvxpy": "unknown", "clarabel": "unknown"}
    summary = {
        "mode": access.mode,
        "upstream_d6_passed": True,
        "oos_data_accessed": False,
        "versions": versions,
        "solver": {
            "name": "CLARABEL",
            "optimal_dates": int(audits["solver_status"].eq("optimal").sum()),
            "success_rate": float(audits["solver_status"].eq("optimal").mean()),
            "fallback_count": fallback_count,
        },
        "objective": {
            "formula": "d^T mu - c * ||w - w0||_1",
            "cost_method": d7["objective_cost_method"],
            "dynamic_impact_in_optimizer": False,
            "dynamic_impact_next_stage": "D8 actual-cost layer via D3",
        },
        "limits": {
            "stock_active": limits.stock_active,
            "industry_active": limits.industry_active,
            "style_active": limits.style_active,
            "constrained_styles": constrained_style_columns,
            "single_sided_turnover": limits.single_sided_turnover,
            "constraint_tolerance": limits.tolerance,
        },
        "signal_coverage": {
            "d6_exposure_start": str(exposures["trade_date"].min().date()),
            "d6_exposure_end": str(exposures["trade_date"].max().date()),
            "optimization_start": str(signal_dates.min().date()),
            "optimization_end": str(signal_dates.max().date()),
            "optimization_dates": int(len(signal_dates)),
            "minimum_mu_coverage": float(audits["mu_coverage"].min()),
            "interpretation": "all D5 walk-forward signal dates inside the 2014-2018 design window",
        },
        "constraint_audit": {
            "maximum_weight_sum_error": float(audits["weight_sum_error"].max()),
            "minimum_weight": float(audits["minimum_weight"].min()),
            "maximum_stock_active": float(audits["maximum_stock_active"].max()),
            "maximum_industry_active": float(audits["maximum_industry_active"].max()),
            "maximum_style_active": float(audits["maximum_style_active"].max()),
            "maximum_single_sided_turnover": float(
                audits["single_sided_turnover"].max()
            ),
            "median_single_sided_turnover": float(
                audits["single_sided_turnover"].median()
            ),
            "maximum_forced_exit_weight": float(audits["forced_exit_weight"].max()),
        },
        "portfolio": {
            "minimum_holdings": int(audits["holdings"].min()),
            "median_holdings": float(audits["holdings"].median()),
            "maximum_holdings": int(audits["holdings"].max()),
        },
        "tracking_error": {
            "estimator": "sklearn.covariance.LedoitWolf",
            "frequency": "daily",
            "annualization_factor": float(d7["covariance_annualization_factor"]),
            "acceptance_statistic": "median",
            "median": te_statistic,
            "mean": float(audits["annualized_tracking_error"].mean()),
            "minimum": float(audits["annualized_tracking_error"].min()),
            "maximum": float(audits["annualized_tracking_error"].max()),
            "acceptance_min": te_min,
            "acceptance_max": te_max,
            "minimum_active_l1_coverage": float(
                audits["covariance_active_l1_coverage"].min()
            ),
            "required_active_l1_coverage": covariance_coverage_min,
            "usage": "report only; not an optimization constraint or tuning target",
        },
        "acceptance": acceptance,
    }

    atomic_parquet(targets, reports_dir / "d7_target_weights.parquet")
    atomic_parquet(audits, reports_dir / "d7_constraint_audit.parquet")
    atomic_parquet(
        factor_exposures, reports_dir / "d7_factor_active_exposures.parquet"
    )
    atomic_parquet(covariance_audit, reports_dir / "d7_covariance_audit.parquet")
    atomic_json(summary, reports_dir / "d7_summary.json")
    write_report(summary, reports_dir)

    print(f"D7 验收：{'达标' if acceptance['passed'] else '未达标'}")
    print(
        f"求解成功：{summary['solver']['optimal_dates']}/{len(signal_dates)}；"
        f"fallback={fallback_count}"
    )
    print(
        f"事前年化 TE 中位数：{te_statistic:.2%}；"
        f"区间 [{te_min:.2%}, {te_max:.2%}]"
    )
    print(f"报告：{reports_dir / 'D7组合优化器报告.md'}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

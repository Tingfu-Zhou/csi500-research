from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.d6_core import (
    FactorExposureError,
    audit_factor_matrix,
    build_industry_dummies,
    compute_raw_style_history,
    estimate_ledoit_wolf_covariance,
    normalize_style,
    process_style_cross_section,
    qlib_to_tushare,
    require_industry_coverage,
    resolve_static_industry_mapping,
    tushare_to_qlib,
    winsorize_mad,
)


def test_stock_code_round_trip() -> None:
    assert tushare_to_qlib("600000.SH") == "SH600000"
    assert tushare_to_qlib("000001.SZ") == "SZ000001"
    assert qlib_to_tushare("SH600000") == "600000.SH"
    with pytest.raises(FactorExposureError, match="无法转换"):
        tushare_to_qlib("600000")


def test_static_industry_mapping_prefers_current_then_most_recent() -> None:
    catalog = pd.DataFrame(
        {
            "l1_code": ["801010.SI", "801020.SI"],
            "industry_name": ["农林牧渔", "采掘"],
        }
    )
    members = pd.DataFrame(
        {
            "l1_code": ["801010.SI", "801020.SI", "801010.SI", "801020.SI"],
            "ts_code": ["600000.SH", "600000.SH", "000001.SZ", "000001.SZ"],
            "is_new": ["N", "Y", "N", "N"],
            "in_date": ["20100101", "20210101", "20100101", "20190101"],
            "out_date": ["20201231", None, "20181231", "20201231"],
        }
    )
    mapping, fixed_catalog = resolve_static_industry_mapping(members, catalog)
    chosen = mapping.set_index("instrument")
    assert chosen.loc["SH600000", "l1_code"] == "801020.SI"
    assert chosen.loc["SZ000001", "l1_code"] == "801020.SI"
    assert fixed_catalog["industry_factor"].is_unique


def test_static_industry_mapping_ignores_non_stock_tushare_codes() -> None:
    catalog = pd.DataFrame(
        {"l1_code": ["801010.SI"], "industry_name": ["农林牧渔"]}
    )
    members = pd.DataFrame(
        {
            "l1_code": ["801010.SI", "801010.SI"],
            "ts_code": ["600000.SH", "T00018.SH"],
            "is_new": ["Y", "Y"],
        }
    )
    mapping, _ = resolve_static_industry_mapping(members, catalog)
    assert mapping["instrument"].tolist() == ["SH600000"]


def test_static_mapping_rejects_conflicting_current_assignments() -> None:
    catalog = pd.DataFrame(
        {"l1_code": ["A", "B"], "industry_name": ["甲", "乙"]}
    )
    members = pd.DataFrame(
        {
            "l1_code": ["A", "B"],
            "ts_code": ["600000.SH", "600000.SH"],
            "is_new": ["Y", "Y"],
        }
    )
    with pytest.raises(FactorExposureError, match="多个当前"):
        resolve_static_industry_mapping(members, catalog)


def test_industry_coverage_fails_loudly() -> None:
    mapping = pd.DataFrame(
        {
            "instrument": ["A"],
            "l1_code": ["I1"],
            "industry_factor": ["industry_I1"],
        }
    )
    with pytest.raises(FactorExposureError, match="缺少 1 只"):
        require_industry_coverage(["A", "B"], mapping)


def test_mad_winsorization_clips_extreme_value() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 3.0, 1000.0])
    result = winsorize_mad(values, 3.0)
    assert result.iloc[-1] < 1000.0
    assert result.iloc[2] == 2.0


def test_style_normalization_is_cap_demeaned_and_ew_scaled_with_neutral_missing() -> None:
    values = pd.Series([-2.0, -0.5, 1.0, 3.0, np.nan])
    market_cap = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result, audit = normalize_style(
        values, market_cap, winsor_mad_multiple=10.0
    )
    cap_weighted_mean = np.average(result, weights=market_cap)
    assert cap_weighted_mean == pytest.approx(0.0, abs=1e-12)
    assert result.std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert result.iloc[-1] == pytest.approx(0.0)
    assert audit["imputed_neutral"] == 1


def style_cross_section() -> pd.DataFrame:
    number = 20
    x = np.linspace(-1.5, 1.5, number)
    return pd.DataFrame(
        {
            "instrument": [f"S{i:03d}" for i in range(number)],
            "market_cap": np.linspace(10.0, 50.0, number),
            "raw_Size": x + 5.0,
            "raw_Beta": np.sin(x) + np.linspace(0.0, 0.3, number),
            "raw_Momentum": x[::-1] ** 2 + np.linspace(0.0, 0.2, number),
            "raw_ResidualVolatility": np.exp(x / 3.0),
            "raw_Liquidity": np.cos(x) + np.linspace(0.0, 0.5, number),
        }
    )


def test_six_style_cross_section_and_nonlinear_size_orthogonality() -> None:
    frame = style_cross_section()
    processed, audit = process_style_cross_section(frame)
    style_columns = [column for column in processed if column.startswith("style_")]
    assert len(style_columns) == 6
    assert len(audit) == 6
    for column in style_columns:
        assert np.average(processed[column], weights=processed["market_cap"]) == pytest.approx(
            0.0, abs=1e-12
        )
        assert processed[column].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    cross_product = np.average(
        processed["style_Size"] * processed["style_NonlinearSize"],
        weights=processed["market_cap"],
    )
    assert cross_product == pytest.approx(0.0, abs=1e-11)


def synthetic_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2017-01-02", periods=320)
    benchmark_return = 0.001 * np.sin(np.arange(len(dates)) / 7.0) + 0.0002
    rng = np.random.default_rng(42)
    rows = []
    basic_rows = []
    for number, beta in enumerate([0.6, 1.0, 1.4, 1.8]):
        noise = rng.normal(0.0, 0.00001 * (number + 1), len(dates))
        returns = beta * benchmark_return + noise
        close = 100.0 * np.cumprod(1.0 + returns)
        instrument = f"S{number}"
        for index, date in enumerate(dates):
            rows.append(
                {"trade_date": date, "instrument": instrument, "close": close[index]}
            )
            basic_rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "total_mv": 1000.0 + number * 200.0 + index,
                    "turnover_rate": 1.0 + number * 0.2 + 0.1 * np.sin(index / 5.0),
                }
            )
    benchmark = pd.DataFrame(
        {"trade_date": dates, "benchmark_return": benchmark_return}
    )
    return pd.DataFrame(rows), benchmark, pd.DataFrame(basic_rows)


def test_raw_style_history_uses_252_day_skip_21_momentum_and_rolling_beta() -> None:
    prices, benchmark, basic = synthetic_history()
    result = compute_raw_style_history(
        prices,
        benchmark,
        basic,
        beta_window=252,
        beta_minimum_observations=126,
        momentum_window=252,
        momentum_skip=21,
        liquidity_window=21,
        liquidity_minimum_observations=5,
    )
    latest = result.loc[result["trade_date"].eq(result["trade_date"].max())].set_index(
        "instrument"
    )
    assert latest.loc["S0", "raw_Beta"] == pytest.approx(0.6, abs=0.04)
    assert latest.loc["S3", "raw_Beta"] == pytest.approx(1.8, abs=0.04)
    prices_s0 = prices.loc[prices["instrument"].eq("S0")].set_index("trade_date")["close"]
    expected_momentum = prices_s0.iloc[-22] / prices_s0.iloc[-253] - 1.0
    assert latest.loc["S0", "raw_Momentum"] == pytest.approx(expected_momentum)
    assert latest["raw_ResidualVolatility"].gt(0).all()
    assert latest["raw_Liquidity"].notna().all()


def test_industry_one_hot_and_full_rank_audit() -> None:
    frame = pd.DataFrame(
        {
            "instrument": [f"S{i}" for i in range(8)],
            "industry_factor": ["industry_I1"] * 4 + ["industry_I2"] * 4,
            "market_cap": np.arange(1.0, 9.0),
            "benchmark_weight": np.arange(1.0, 9.0) / 36.0,
        }
    )
    first, _ = normalize_style(
        pd.Series([-3, -1, 0, 2, -2, 1, 3, 5], dtype=float),
        frame["market_cap"],
        winsor_mad_multiple=10,
    )
    second, _ = normalize_style(
        pd.Series([1, -2, 4, -3, 5, -1, 0, 2], dtype=float),
        frame["market_cap"],
        winsor_mad_multiple=10,
    )
    frame["style_A"] = first
    frame["style_B"] = second
    catalog = pd.DataFrame(
        {
            "l1_code": ["I1", "I2"],
            "industry_factor": ["industry_I1", "industry_I2"],
        }
    )
    matrix, industry_columns = build_industry_dummies(frame, catalog)
    audit = audit_factor_matrix(
        matrix, industry_columns, ["style_A", "style_B"]
    )
    assert audit["full_rank"]
    assert audit["matrix_rank"] == 4
    assert audit["industry_row_sum_max_error"] == 0.0
    assert audit["benchmark_industry_exposure_max_error"] == pytest.approx(0.0)
    assert audit["passed"]


def test_full_rank_ignores_globally_defined_but_inactive_industry_column() -> None:
    frame = pd.DataFrame(
        {
            "instrument": [f"S{i}" for i in range(8)],
            "industry_factor": ["industry_I1"] * 4 + ["industry_I2"] * 4,
            "market_cap": np.arange(1.0, 9.0),
            "benchmark_weight": np.repeat(0.125, 8),
        }
    )
    frame["style_A"], _ = normalize_style(
        pd.Series([-3, -1, 0, 2, -2, 1, 3, 5], dtype=float),
        frame["market_cap"],
        winsor_mad_multiple=10,
    )
    frame["style_B"], _ = normalize_style(
        pd.Series([1, -2, 4, -3, 5, -1, 0, 2], dtype=float),
        frame["market_cap"],
        winsor_mad_multiple=10,
    )
    catalog = pd.DataFrame(
        {
            "l1_code": ["I1", "I2", "I3"],
            "industry_factor": ["industry_I1", "industry_I2", "industry_I3"],
        }
    )
    matrix, industry_columns = build_industry_dummies(frame, catalog)
    audit = audit_factor_matrix(
        matrix, industry_columns, ["style_A", "style_B"]
    )
    assert audit["global_columns"] == 5
    assert audit["global_matrix_rank"] == 4
    assert audit["active_industry_columns"] == 2
    assert audit["columns"] == 4
    assert audit["matrix_rank"] == 4
    assert audit["inactive_industry_columns"] == "industry_I3"
    assert audit["full_rank"]
    assert audit["passed"]


def test_ledoit_wolf_covariance_is_psd_and_excludes_short_history() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-02", periods=260)
    rows = []
    for instrument, start in [("A", 0), ("B", 0), ("C", 150)]:
        for date, value in zip(dates[start:], rng.normal(0, 0.01, len(dates) - start)):
            rows.append(
                {"trade_date": date, "instrument": instrument, "return": value}
            )
    estimate = estimate_ledoit_wolf_covariance(
        pd.DataFrame(rows),
        as_of_date=dates[-1],
        instruments=["A", "B", "C"],
        window=252,
        minimum_observations=126,
    )
    assert estimate.instruments == ("A", "B")
    assert estimate.excluded_instruments == ("C",)
    assert estimate.covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(estimate.covariance).min() >= -1e-12
    assert 0.0 <= estimate.shrinkage <= 1.0

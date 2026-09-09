"""Regression checks for the narrow D12 direct-price evidence collector."""
import pandas as pd
import pytest

from scripts.d12_collect_price_reference import build_summary, normalize


def test_direct_reference_exposes_factor_jump_separately_from_raw_return():
    dates = ["20250221", "20250224"]
    daily = pd.DataFrame({
        "ts_code": ["302132.SZ"] * 2,
        "trade_date": dates,
        "open": [65.0, 65.0], "high": [66.0, 66.0], "low": [64.0, 64.0],
        "close": [65.8, 64.8], "pre_close": [65.0, 65.8],
        "change": [.8, -1.0], "pct_chg": [1.23, -1.52],
        "vol": [1., 1.], "amount": [1., 1.],
    })
    factors = pd.DataFrame({
        "ts_code": ["302132.SZ"] * 2,
        "trade_date": dates,
        "adj_factor": [1., 6.],
    })
    basic = pd.DataFrame({
        "ts_code": ["302132.SZ"] * 2,
        "trade_date": dates,
        "total_share": [1., 6.], "float_share": [1., 1.], "free_share": [1., 1.],
        "total_mv": [1., 6.], "circ_mv": [1., 1.],
    })
    result = build_summary(daily, factors, basic).iloc[-1]
    assert result.adj_factor_ratio == pytest.approx(6.)
    assert result.adjusted_return_from_direct_fields == pytest.approx(64.8 * 6 / 65.8 - 1)
    assert result.pct_chg == pytest.approx(-1.52)


def test_direct_reference_rejects_conflicting_duplicate_keys():
    frame = pd.DataFrame({
        "ts_code": ["302132.SZ", "302132.SZ"],
        "trade_date": ["20250224", "20250224"],
        "close": [64.8, 64.9],
    })
    with pytest.raises(RuntimeError, match="重复"):
        normalize(frame, ["ts_code", "trade_date", "close"], "daily")

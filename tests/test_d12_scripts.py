"""D12 day-1 scripts: the cost reconciliation and the repository inventory helpers.

The reconciliation is exercised against a fills table built by calling `trade_cost`
itself, so a change to the cost model breaks this test rather than passing silently.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d12_impact_reconciliation import reconcile
from src.d3_core import CostModel, trade_cost
from src.d12_repo import (manifest_coverage, pending_code_changes, protected_blob_drift,
                          registered_sha256, required_inputs)

CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text())


def synthetic_trades(rows=300, seed=4):
    """Fills shaped exactly like `src.d11_core.execute_orders` writes them."""
    rng = np.random.default_rng(seed)
    model = CostModel.from_config(CONFIG)
    # Both stamp-duty regimes, so the date-dependent rate is exercised.
    dates = pd.to_datetime(rng.choice(["2021-05-11", "2023-03-14", "2024-06-18", "2026-02-27"], rows))
    records = []
    for date in dates:
        nav = float(rng.uniform(1e7, 3e7))
        sigma, amount = float(rng.uniform(0.005, 0.09)), float(rng.uniform(3e7, 2e9))
        weight = float(rng.uniform(-0.01, 0.01)) if rng.random() > 0.1 else 0.0
        notional = np.clip(weight * nav, -0.04 * amount, 0.04 * amount)
        costs = trade_cost(notional, sigma, amount, date, model)
        records.append({"execution_date": date, "signal_nav": nav,
                        "executed_trade_amount": float(notional), **costs})
    return pd.DataFrame(records), model


def test_reconcile_reproduces_every_charged_cost_component():
    trades, model = synthetic_trades()
    result = reconcile(trades, model)
    assert result["identities_hold"]
    assert result["identities_max_relative_deviation"] < 1e-12
    assert result["rows"] == len(trades)
    assert 0 < result["filled_rows"] <= result["rows"]


def test_reconcile_recovers_the_impact_coefficient_that_produced_the_fills():
    """c = k * sigma * sqrt(NAV / A) must come back out of the ledger alone."""
    trades, model = synthetic_trades()
    traded = trades["executed_trade_amount"].abs().gt(0)
    weight = (trades["executed_trade_amount"].abs() / trades["signal_nav"])[traded]
    recovered = (trades["impact_rate"][traded] / np.sqrt(weight))
    result = reconcile(trades, model)
    assert result["recovered_coefficient_c"]["count"] == int(traded.sum())
    assert result["recovered_coefficient_c"]["mean"] == pytest.approx(recovered.mean(), rel=1e-12)
    for quantile, value in result["recovered_coefficient_c"]["quantiles"].items():
        assert value == pytest.approx(recovered.quantile(float(quantile)), rel=1e-12)
    # The headline "a 1% trade costs this much" is the same number, scaled.
    for quantile, basis_points in result["expected_impact_of_a_one_percent_trade_bps"].items():
        assert basis_points == pytest.approx(
            recovered.quantile(float(quantile)) * 0.01 ** 1.5 * 1e4, rel=1e-12)


def test_reconcile_separates_real_trades_from_optimiser_dust():
    """freeze_orders keeps orders down to 1e-12 of NAV, so most rows are not trades."""
    from scripts.d12_impact_reconciliation import DUST_THRESHOLD

    trades, model = synthetic_trades(rows=200, seed=9)
    # Half the rows become dust: a notional far below the threshold, still non-zero.
    dust = trades.index[::2]
    trades.loc[dust, "executed_trade_amount"] = trades.loc[dust, "signal_nav"] * 1e-10
    for column in ["commission_cost", "transfer_fee_cost", "stamp_duty_cost", "impact_cost"]:
        trades.loc[dust, column] = 0.0
    trades.loc[dust, ["total_cost", "impact_rate"]] = 0.0
    weight = trades["executed_trade_amount"].abs() / trades["signal_nav"]
    expected = int((weight > DUST_THRESHOLD).sum())  # excludes dust and the untraded rows
    concentration = reconcile(trades, model)["traded_weight_concentration"]
    assert concentration["dust_threshold"] == DUST_THRESHOLD
    assert 0 < expected < len(trades) - len(dust)  # some rows never traded at all
    assert concentration["rows_above_threshold"] == expected
    assert concentration["share_of_rows"] == pytest.approx(expected / len(trades), rel=1e-12)
    # Dust carries a vanishing share of the traded weight even at half the rows.
    assert concentration["share_of_traded_weight_above_threshold"] > 0.9999
    assert concentration["decisions"] == trades["execution_date"].nunique()
    assert concentration["notional_weighted_impact_rate_bp"] > 0


def test_reconcile_reports_saved_totals_and_notices_a_broken_ledger():
    trades, model = synthetic_trades()
    saved = {"impact_cost": float(trades["impact_cost"].sum()), "total_cost": 1.0}
    result = reconcile(trades, model, saved)
    assert result["against_saved_summary"]["impact_cost"]["difference"] == pytest.approx(0, abs=1e-6)
    assert result["against_saved_summary"]["total_cost"]["difference"] != 0
    broken = trades.copy()
    broken.loc[broken.index[0], "impact_cost"] += 1.0
    assert not reconcile(broken, model)["identities_hold"]


def test_reconcile_refuses_a_frame_missing_a_cost_column():
    trades, model = synthetic_trades(rows=20)
    with pytest.raises(KeyError, match="impact_rate"):
        reconcile(trades.drop(columns=["impact_rate"]), model)


def test_manifest_coverage_splits_declared_from_present(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"output_sha256": {"a.json": "x", "b.parquet": "y"}}))
    (run / "a.json").write_text("{}")
    coverage = manifest_coverage(run)
    assert coverage == {"declared": 2, "manifest_present": True, "run_dir_present": True,
                        "missing": ["b.parquet"]}
    absent = manifest_coverage(tmp_path / "nope")
    assert absent["declared"] == 0 and absent["manifest_present"] is False


def test_the_accepted_d1_to_d10_files_are_still_byte_identical():
    baseline = json.loads((ROOT / "config/d11_baseline.json").read_text())
    assert protected_blob_drift(ROOT, baseline) == []
    assert len(baseline["protected_blobs"]) == 140


def test_registered_digests_still_match_where_the_file_is_present():
    from src.d12_repo import sha256

    for relative, expected in registered_sha256(ROOT).items():
        path = ROOT / relative
        if path.is_file():  # the parquet is gitignored and absent from a fresh clone
            assert sha256(path) == expected, relative


def test_pending_code_changes_only_looks_at_code_directories():
    changes = pending_code_changes(ROOT)
    assert all(path.split("/")[0] in {"src", "scripts", "tests", "config"} for path in changes)


def test_required_inputs_are_all_relative_paths():
    paths = required_inputs(CONFIG)
    assert len(paths) == 36
    assert all(not path.is_absolute() for path in paths)

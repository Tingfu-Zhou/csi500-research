from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d2_pipeline import (  # noqa: E402
    load_intervals,
    load_yearly as load_d2_yearly,
    make_st_pairs,
    qlib_features,
)
from scripts.d3_pipeline import (  # noqa: E402
    add_exchange_flags,
    add_ex_ante_volatility,
    load_qlib_market,
    load_yearly as load_d3_yearly,
    prepare_markets,
    summarize_execution,
)
from scripts.d4_pipeline import (  # noqa: E402
    fit_default_lightgbm,
    load_alpha158_features,
    load_vwap,
)
from scripts.d6_pipeline import (  # noqa: E402
    assemble_exposures,
    audit_weekly_matrices,
    load_benchmark_returns,
    load_close_prices,
    load_daily_basic,
    require_industry_files,
    usable_style_dates,
)
from scripts.d7_pipeline import (  # noqa: E402
    build_execution_price_matrix,
    holding_returns_between,
    make_execution_schedule,
    validate_d7_config,
)
from scripts.d8_pipeline import (  # noqa: E402
    audit_all_layer_targets,
    build_frozen_d7_pretrade_path,
    build_stock_returns,
    calculate_layer_tracking_error,
    generate_optimized_layer_targets,
    optimizer_limits_for_layer,
    run_loss_ladder_backtests,
)
from src.d2_core import (  # noqa: E402
    add_backcast_snapshots,
    apply_membership_replacements,
    build_daily_weights,
    compare_snapshot_membership,
    expand_membership,
    is_close,
    normalize_snapshots,
    to_qlib_symbol,
    trade_day_age,
)
from src.d3_core import CostModel, run_vwap_backtest  # noqa: E402
from src.d4_core import (  # noqa: E402
    build_forward_excess_labels,
    rank_ic_series,
    select_weekly_sample_dates,
    summarize_ic,
)
from src.d5_core import (  # noqa: E402
    apply_return_calibration,
    build_expanding_year_folds,
    fit_return_calibration,
    rank_gaussianize,
    split_walk_forward_samples,
)
from src.d6_core import (  # noqa: E402
    STYLE_FACTORS,
    compute_raw_style_history,
    estimate_ledoit_wolf_covariance,
    require_industry_coverage,
)
from src.d7_core import (  # noqa: E402
    PortfolioOptimizationError,
    annualized_tracking_error,
    drift_pretrade_weights,
    optimize_long_only_active_portfolio,
    symmetric_l1_cost_rate,
)
from src.d8_core import (  # noqa: E402
    LAYER_NAMES,
    build_loss_deltas,
    make_ideal_execution_market,
    make_top_n_equal_targets,
    summarize_backtest_performance,
)
from src.d9_core import (  # noqa: E402
    OOSIntegrityError,
    OOSThresholds,
    apply_identifier_alias_removals,
    apply_snapshot_instrument_bridges,
    assert_frozen_config_unchanged,
    audit_minimum_turnover_feasibility,
    evaluate_oos_thresholds,
    optimize_long_only_active_portfolio_v2,
    summarize_monthly_performance,
    validate_membership_recovery_failure,
    validate_oos_fold_records,
    validate_v2_source_failures,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_frozen_repository(config: dict) -> dict[str, object]:
    """Prove that D9 descends from the immutable freeze without changing D1–D8."""

    d9 = config["d9"]
    expected = str(d9["frozen_oos_commit"])
    branch = git_output("branch", "--show-current")
    if branch != str(d9["required_branch"]):
        raise OOSIntegrityError(
            f"D9 只能在 {d9['required_branch']} 分支运行，当前为 {branch!r}"
        )
    tag_commit = git_output("rev-parse", "frozen-oos^{commit}")
    if tag_commit != expected:
        raise OOSIntegrityError(
            f"frozen-oos 指向 {tag_commit}，预注册冻结提交为 {expected}"
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tracked_status = git_output("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise OOSIntegrityError("运行 D9 前已跟踪文件必须无本地未提交改动")
    changed_text = git_output("diff", "--name-only", "-z", f"{expected}..HEAD")
    changed = sorted(filter(None, changed_text.split("\0")))
    allowed = sorted(str(value) for value in d9["allowed_changes_since_freeze"])
    unexpected = sorted(set(changed) - set(allowed))
    if unexpected:
        raise OOSIntegrityError(
            "冻结后出现非 D9 文件改动：" + ", ".join(unexpected)
        )
    frozen_text = git_output("show", f"{expected}:config/config.yaml")
    frozen_config = yaml.safe_load(frozen_text)
    assert_frozen_config_unchanged(config, frozen_config)
    head = git_output("rev-parse", "HEAD")
    return {
        "branch": branch,
        "frozen_oos_commit": tag_commit,
        "d9_head_commit": head,
        "changed_paths_since_freeze": changed,
        "allowed_paths_only": True,
        "tracked_worktree_clean": True,
        "frozen_config_unchanged": True,
    }


def require_upstream_artifacts(config: dict) -> dict[str, dict]:
    reports = ROOT / config["data"]["reports_dir"]
    payloads: dict[str, dict] = {}
    for day in range(2, 9):
        filename = "d2_qc_summary.json" if day == 2 else f"d{day}_summary.json"
        path = reports / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path.relative_to(ROOT)}")
        payloads[f"d{day}"] = json.loads(path.read_text(encoding="utf-8"))
    for day in [2, 3, 4, 5, 6, 8]:
        if payloads[f"d{day}"].get("acceptance", {}).get("passed") is not True:
            raise OOSIntegrityError(f"冻结的 D{day} 未通过，不得运行 D9")
    d7_acceptance = payloads["d7"].get("acceptance", {})
    structural = [
        "upstream_d6",
        "all_available_design_signal_dates",
        "solver_success_100pct",
        "fallback_zero",
        "fully_invested",
        "long_only",
        "stock_active",
        "industry_active",
        "style_active",
        "turnover",
        "covariance_coverage",
    ]
    failed = [key for key in structural if d7_acceptance.get(key) is not True]
    if failed:
        raise OOSIntegrityError("冻结的 D7 结构验收失败：" + ", ".join(failed))
    return payloads


def required_data_paths(config: dict, mode: str) -> list[Path]:
    raw = ROOT / config["data"]["tushare_raw_dir"]
    sealed = ROOT / config["data"]["tushare_sealed_dir"]
    design = ROOT / config["data"]["processed_design_dir"]
    reports = ROOT / config["data"]["reports_dir"]
    d6 = config["d6"]
    paths = [
        ROOT / config["data"]["qlib_provider_uri"].lstrip("/")
        if not Path(config["data"]["qlib_provider_uri"]).is_absolute()
        else Path(config["data"]["qlib_provider_uri"]),
        design / "benchmark_weights.parquet",
        design / "tradability_flags.parquet",
        ROOT / d6["industry_mapping_file"],
        ROOT / d6["industry_catalog_file"],
        reports / "d5_walk_forward_predictions.parquet",
    ]
    core = [
        "trade_cal.parquet",
        "index_daily_H00905_CSI.parquet",
        "index_weight_000905_SH.parquet",
        "namechange.parquet",
        "stock_basic.parquet",
    ]
    if mode == "design":
        paths.extend(sealed / "full_snapshot" / name for name in core)
        paths.extend(sealed / f"{prefix}_{year}.parquet" for prefix in ["daily_basic", "stk_limit", "suspend_d"] for year in range(2019, 2027))
    elif mode == "oos":
        paths.extend(raw / name for name in core)
        paths.extend(raw / f"{prefix}_{year}.parquet" for prefix in ["daily_basic", "stk_limit", "suspend_d"] for year in range(2019, 2027))
    else:
        raise OOSIntegrityError(f"未知窗口模式：{mode}")
    return paths


def preflight(
    config: dict,
    *,
    membership_recovery: bool = False,
    v2_stage: str | None = None,
) -> tuple[dict[str, object], dict[str, dict]]:
    if membership_recovery and v2_stage is not None:
        raise ValueError("成分恢复与 D9 V2 模式不能同时启用")
    if v2_stage not in {None, "audit", "run"}:
        raise ValueError(f"未知 D9 V2 阶段：{v2_stage}")
    repository = verify_frozen_repository(config)
    upstream = require_upstream_artifacts(config)
    access = ResearchDataAccess(ROOT)
    manifest = ROOT / config["d9"]["run_manifest"]
    summary = ROOT / config["data"]["reports_dir"] / "d9_summary.json"
    recovery_audit: dict[str, object] | None = None
    v2_audit: dict[str, object] | None = None
    if membership_recovery:
        recovery = config["d9"]["membership_recovery"]
        recovery_manifest = ROOT / recovery["recovery_manifest"]
        if not manifest.exists():
            raise OOSIntegrityError("缺少原始 D9 失败清单，不得受控恢复")
        if summary.exists() or recovery_manifest.exists():
            raise OOSIntegrityError("检测到 D9 结果或恢复清单，拒绝再次恢复")
        original_payload = json.loads(manifest.read_text(encoding="utf-8"))
        recovery_audit = validate_membership_recovery_failure(
            original_payload, recovery["original_failure"]
        )
        recovery_audit.update(
            {
                "original_manifest": str(manifest.relative_to(ROOT)),
                "original_manifest_sha256": sha256(manifest),
                "recovery_manifest": str(recovery_manifest.relative_to(ROOT)),
                "original_manifest_preserved": True,
                "scope": "benchmark_membership_metadata_only",
            }
        )
    elif v2_stage is not None:
        v2 = config["d9"]["v2"]
        recovery_manifest = ROOT / config["d9"]["membership_recovery"][
            "recovery_manifest"
        ]
        if not manifest.exists() or not recovery_manifest.exists():
            raise OOSIntegrityError("D9 V2 要求原始及恢复失败清单同时存在")
        if summary.exists():
            raise OOSIntegrityError("检测到原版 D9 完整结果，不得启动事后修订 V2")
        original_hash = sha256(manifest)
        recovery_hash = sha256(recovery_manifest)
        if original_hash != str(v2["original_manifest_sha256"]):
            raise OOSIntegrityError("D9 原始失败清单 SHA256 与 V2 登记不一致")
        if recovery_hash != str(v2["recovery_manifest_sha256"]):
            raise OOSIntegrityError("D9 恢复失败清单 SHA256 与 V2 登记不一致")
        original_payload = json.loads(manifest.read_text(encoding="utf-8"))
        recovery_payload = json.loads(
            recovery_manifest.read_text(encoding="utf-8")
        )
        v2_audit = validate_v2_source_failures(
            original_payload,
            recovery_payload,
            expected_original=config["d9"]["membership_recovery"][
                "original_failure"
            ],
            expected_recovery=v2["recovery_failure"],
        )
        v2_audit.update(
            {
                "original_manifest": str(manifest.relative_to(ROOT)),
                "original_manifest_sha256": original_hash,
                "recovery_manifest": str(recovery_manifest.relative_to(ROOT)),
                "recovery_manifest_sha256": recovery_hash,
                "source_manifests_preserved": True,
            }
        )
        feasibility_path = ROOT / v2["feasibility_audit"]
        feasibility_dates_path = ROOT / v2["feasibility_dates"]
        v2_manifest_path = ROOT / v2["run_manifest"]
        v2_summary_path = ROOT / v2["summary"]
        if v2_stage == "audit":
            existing = [
                path for path in [
                    feasibility_path,
                    v2_manifest_path,
                    v2_summary_path,
                ]
                if path.exists()
            ]
            if existing:
                raise OOSIntegrityError(
                    "D9 V2 审计或运行记录已存在，拒绝覆盖："
                    + ", ".join(str(path.relative_to(ROOT)) for path in existing)
                )
        else:
            if not feasibility_path.exists() or not feasibility_dates_path.exists():
                raise OOSIntegrityError("D9 V2 正式运行前必须先完成可行性审计")
            if v2_manifest_path.exists() or v2_summary_path.exists():
                raise OOSIntegrityError("D9 V2 已有运行记录或结果，拒绝再次启动")
            feasibility = json.loads(
                feasibility_path.read_text(encoding="utf-8")
            )
            required = {
                "status": "passed",
                "classification": "post_oos_methodology_amendment",
                "original_manifest_sha256": original_hash,
                "recovery_manifest_sha256": recovery_hash,
                "v2_head_commit": repository["d9_head_commit"],
                "config_sha256": sha256(ROOT / "config/config.yaml"),
                "portfolio_performance_generated": False,
            }
            mismatch = {
                key: {"expected": value, "actual": feasibility.get(key)}
                for key, value in required.items()
                if feasibility.get(key) != value
            }
            if mismatch:
                raise OOSIntegrityError(
                    "D9 V2 可行性审计凭据不匹配：" + str(mismatch)
                )
            if feasibility.get("v2_all_dates_feasible") is not True:
                raise OOSIntegrityError("D9 V2 可行性审计未证明全部日期可解")
            if feasibility.get("frozen_first_failure", {}).get(
                "frozen_rule_feasible"
            ) is not False:
                raise OOSIntegrityError("D9 V2 审计没有复现冻结换手规则不可行")
            if feasibility.get("feasibility_dates_sha256") != sha256(
                feasibility_dates_path
            ):
                raise OOSIntegrityError("D9 V2 分日可行性审计文件哈希不匹配")
            v2_audit["feasibility_audit"] = str(
                feasibility_path.relative_to(ROOT)
            )
            v2_audit["feasibility_audit_sha256"] = sha256(feasibility_path)
            v2_audit["feasibility_dates_sha256"] = sha256(
                feasibility_dates_path
            )
    elif manifest.exists() or summary.exists():
        raise OOSIntegrityError(
            "检测到 D9 运行记录或结果；样本外只允许运行一次，拒绝再次启动。"
            "若且仅若这是已登记的 35 日成分失败，请按运行说明使用受控恢复命令"
        )
    missing = [
        str(path)
        for path in required_data_paths(config, access.mode)
        if not path.exists()
    ]
    provider = Path(config["data"]["qlib_provider_uri"])
    membership = provider / "instruments/csi500.txt"
    if not membership.exists():
        missing.append(str(membership))
    if missing:
        hint = (
            "\n当前仍为 design 模式；请先运行 "
            "python scripts/pull_d9_oos.py，将缺失的 OOS 原始文件直接写入 sealed。"
            if access.mode == "design"
            else ""
        )
        raise FileNotFoundError(
            "D9 预检缺少文件：\n- " + "\n- ".join(missing) + hint
        )
    try:
        import clarabel
        import cvxpy as cp
        import lightgbm
        import qlib

        if "CLARABEL" not in cp.installed_solvers():
            raise ImportError("cvxpy 未发现 CLARABEL")
    except ImportError as exc:
        raise RuntimeError(f"D9 依赖预检失败：{exc}") from exc
    installed_versions = {
        "qlib": str(qlib.__version__),
        "lightgbm": str(lightgbm.__version__),
        "cvxpy": str(cp.__version__),
        "clarabel": str(clarabel.__version__),
    }
    frozen_versions = {
        "qlib": str(upstream["d4"]["versions"]["qlib"]),
        "lightgbm": str(upstream["d5"]["versions"]["lightgbm"]),
        "cvxpy": str(upstream["d7"]["versions"]["cvxpy"]),
        "clarabel": str(upstream["d7"]["versions"]["clarabel"]),
    }
    version_mismatches = {
        name: {"installed": installed_versions[name], "frozen": frozen}
        for name, frozen in frozen_versions.items()
        if installed_versions[name] != frozen
    }
    if version_mismatches:
        raise RuntimeError(
            "D9 依赖版本与冻结报告不一致："
            + json.dumps(version_mismatches, ensure_ascii=False)
        )
    return (
        {
            **repository,
            "window_mode": access.mode,
            "run_manifest_absent": not manifest.exists(),
            "required_files_present": True,
            "dependencies_present": True,
            "dependency_versions": installed_versions,
            "dependency_versions_match_freeze": True,
            "membership_recovery": recovery_audit,
            "v2_source_failures": v2_audit,
            "v2_stage": v2_stage,
        },
        upstream,
    )


def load_open_calendar(
    access: ResearchDataAccess, config: dict, *, end_date
) -> pd.DatetimeIndex:
    raw = Path(config["data"]["tushare_raw_dir"])
    frame = access.read_parquet(
        raw / "trade_cal.parquet", date_col="cal_date", end_date=end_date
    )
    frame["cal_date"] = parse_date_series(frame["cal_date"])
    calendar = pd.DatetimeIndex(
        frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"]
    ).normalize().sort_values().unique()
    if calendar.empty:
        raise ValueError("D9 交易日历为空")
    return calendar


def assert_design_weight_parity(full_weights: pd.DataFrame, config: dict) -> None:
    path = ROOT / config["data"]["processed_design_dir"] / "benchmark_weights.parquet"
    frozen = pd.read_parquet(path)
    frozen["trade_date"] = parse_date_series(frozen["trade_date"])
    design_end = pd.Timestamp(config["windows"]["design_end"])
    rebuilt = full_weights.loc[full_weights["trade_date"].le(design_end)].copy()
    columns = ["trade_date", "instrument", "weight", "snapshot_date", "snapshot_origin"]
    left = frozen[columns].sort_values(["trade_date", "instrument"]).reset_index(drop=True)
    right = rebuilt[columns].sort_values(["trade_date", "instrument"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left,
        right,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-14,
    )


def validate_benchmark_membership_metadata(
    access: ResearchDataAccess,
    config: dict,
    calendar: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove observed and synthetic membership sets before reading any prices."""

    windows = config["windows"]
    design_start = pd.Timestamp(windows["design_start"])
    oos_end = pd.Timestamp(windows["oos_end"])
    raw = Path(config["data"]["tushare_raw_dir"])
    frozen_replacements = config["universe"].get("membership_replacements", [])
    recovery = config["d9"].get("membership_recovery", {})
    recovery_replacements = recovery.get("membership_replacements", [])
    replacements = [*frozen_replacements, *recovery_replacements]

    intervals = load_intervals(config, oos_end)
    intervals, _ = apply_membership_replacements(intervals, frozen_replacements)
    intervals, _ = apply_membership_replacements(intervals, recovery_replacements)
    intervals, alias_audit = apply_identifier_alias_removals(
        intervals, recovery.get("identifier_alias_removals", [])
    )
    source = access.read_parquet(
        raw / "index_weight_000905_SH.parquet",
        date_col="trade_date",
        end_date=oos_end,
    )
    source["trade_date"] = parse_date_series(source["trade_date"])
    source["instrument"] = source["con_code"].map(to_qlib_symbol)
    source["weight"] = pd.to_numeric(source["weight"], errors="raise")
    snapshots = normalize_snapshots(
        source[["trade_date", "instrument", "weight"]].drop_duplicates()
    )
    snapshots["snapshot_origin"] = "tushare_index_weight"
    bootstrap = snapshots.loc[
        snapshots["trade_date"].le(design_start), "trade_date"
    ].max()
    if pd.isna(bootstrap):
        raise ValueError("D9 缺少设计窗口前的启动权重快照")
    weight_calendar = calendar[
        (calendar >= pd.Timestamp(bootstrap)) & (calendar <= oos_end)
    ]
    membership = expand_membership(intervals, weight_calendar)
    snapshots = snapshots.loc[
        snapshots["trade_date"].between(weight_calendar.min(), oos_end)
    ].copy()
    observed_audit = compare_snapshot_membership(membership, snapshots)
    observed_mismatches = int(observed_audit["status"].ne("ok").sum())
    if observed_mismatches:
        raise ValueError(
            f"受控恢复后仍有 {observed_mismatches} 个真实权重快照日不一致"
        )

    synthetic: list[pd.DataFrame] = []
    synthetic_dates: list[pd.Timestamp] = []
    for number, replacement in enumerate(replacements, start=1):
        effective = pd.Timestamp(replacement["effective_start"]).normalize()
        anchor = pd.Timestamp(replacement["anchor_snapshot_date"]).normalize()
        outgoing = str(replacement["remove_instrument"]).upper()
        incoming = str(replacement["add_instrument"]).upper()
        if snapshots["trade_date"].eq(effective).any():
            raise ValueError(f"{effective.date()} 已有真实权重快照，无需反推")
        anchor_rows = snapshots.loc[snapshots["trade_date"].eq(anchor)].copy()
        anchor_set = set(anchor_rows["instrument"].astype(str))
        if not anchor_set or incoming not in anchor_set or outgoing in anchor_set:
            raise ValueError(f"第 {number} 条修正的锚点快照与方向不一致")
        anchor_rows["trade_date"] = effective
        anchor_rows["snapshot_origin"] = "backcast_membership_correction"
        synthetic.append(anchor_rows)
        synthetic_dates.append(effective)
    metadata_snapshots = pd.concat([snapshots, *synthetic], ignore_index=True)
    metadata_snapshots, bridge_audit = apply_snapshot_instrument_bridges(
        metadata_snapshots,
        recovery.get("synthetic_snapshot_bridges", []),
    )
    synthetic_audit = compare_snapshot_membership(
        membership,
        metadata_snapshots.loc[
            metadata_snapshots["trade_date"].isin(synthetic_dates)
        ],
    )
    synthetic_mismatches = int(synthetic_audit["status"].ne("ok").sum())
    if synthetic_mismatches:
        raise ValueError(
            f"受控恢复后仍有 {synthetic_mismatches} 个合成调样日不一致"
        )
    return {
        "prices_or_returns_read": False,
        "observed_snapshot_dates": int(len(observed_audit)),
        "observed_snapshot_mismatch_days": observed_mismatches,
        "synthetic_event_dates": int(len(synthetic_audit)),
        "synthetic_event_mismatch_days": synthetic_mismatches,
        "identifier_alias_rules": int(len(alias_audit)),
        "synthetic_snapshot_bridges": int(len(bridge_audit)),
        "passed": True,
    }


def build_full_benchmark_weights(
    access: ResearchDataAccess,
    config: dict,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, object]]:
    windows = config["windows"]
    design_start = pd.Timestamp(windows["design_start"])
    oos_end = pd.Timestamp(windows["oos_end"])
    raw = Path(config["data"]["tushare_raw_dir"])
    intervals = load_intervals(config, oos_end)
    frozen_replacements = config["universe"].get("membership_replacements", [])
    recovery = config["d9"].get("membership_recovery", {})
    recovery_replacements = recovery.get("membership_replacements", [])
    intervals, frozen_correction_audit = apply_membership_replacements(
        intervals, frozen_replacements
    )
    intervals, recovery_correction_audit = apply_membership_replacements(
        intervals, recovery_replacements
    )
    intervals, alias_audit = apply_identifier_alias_removals(
        intervals, recovery.get("identifier_alias_removals", [])
    )
    source = access.read_parquet(
        raw / "index_weight_000905_SH.parquet",
        date_col="trade_date",
        end_date=oos_end,
    )
    source["trade_date"] = parse_date_series(source["trade_date"])
    source["instrument"] = source["con_code"].map(to_qlib_symbol)
    source["weight"] = pd.to_numeric(source["weight"], errors="raise")
    snapshots = normalize_snapshots(
        source[["trade_date", "instrument", "weight"]].drop_duplicates()
    )
    snapshots["snapshot_origin"] = "tushare_index_weight"
    bootstrap = snapshots.loc[
        snapshots["trade_date"].le(design_start), "trade_date"
    ].max()
    if pd.isna(bootstrap):
        raise ValueError("D9 缺少设计窗口前的启动权重快照")
    weight_calendar = calendar[
        (calendar >= pd.Timestamp(bootstrap)) & (calendar <= oos_end)
    ]
    membership = expand_membership(intervals, weight_calendar)
    snapshots = snapshots.loc[
        snapshots["trade_date"].between(weight_calendar.min(), oos_end)
    ].copy()
    membership_audit = compare_snapshot_membership(membership, snapshots)
    mismatches = int(membership_audit["status"].ne("ok").sum())
    if mismatches:
        raise ValueError(f"D9 权重快照与 PIT 成分区间有 {mismatches} 个不一致日")
    instruments = sorted(membership["instrument"].astype(str).unique())
    close_long = load_close_prices(
        access, instruments, weight_calendar.min(), oos_end
    )
    close = close_long.pivot(
        index="trade_date", columns="instrument", values="close"
    )
    snapshots = add_backcast_snapshots(
        snapshots,
        close,
        [*frozen_replacements, *recovery_replacements],
    )
    snapshots, bridge_audit = apply_snapshot_instrument_bridges(
        snapshots,
        recovery.get("synthetic_snapshot_bridges", []),
    )
    weights = build_daily_weights(
        weight_calendar,
        membership,
        snapshots,
        close,
        design_start,
        oos_end,
    )
    assert_design_weight_parity(weights, config)
    sums = weights.groupby("trade_date")["weight"].sum()
    counts = weights.groupby("trade_date")["instrument"].nunique()
    if not counts.eq(500).all():
        raise ValueError("D9 日度 PIT 成分数并非始终为 500")
    return weights, {
        "start": str(weights["trade_date"].min().date()),
        "end": str(weights["trade_date"].max().date()),
        "trade_days": int(weights["trade_date"].nunique()),
        "rows": int(len(weights)),
        "constituents_min": int(counts.min()),
        "constituents_max": int(counts.max()),
        "maximum_weight_sum_error": float((sums - 1.0).abs().max()),
        "snapshot_mismatch_days": mismatches,
        "frozen_membership_corrections": frozen_correction_audit.to_dict(
            orient="records"
        ),
        "d9_membership_replacements": recovery_correction_audit.to_dict(
            orient="records"
        ),
        "d9_identifier_alias_removals": alias_audit,
        "d9_synthetic_snapshot_bridges": bridge_audit,
        "membership_recovery_rules": int(
            len(recovery_correction_audit) + len(alias_audit) + len(bridge_audit)
        ),
        "design_parity_passed": True,
    }


def build_oos_tradability(
    access: ResearchDataAccess,
    config: dict,
    weights: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, object]]:
    oos_start = pd.Timestamp(config["windows"]["oos_start"])
    oos_end = pd.Timestamp(config["windows"]["oos_end"])
    history_start = oos_start - pd.Timedelta(days=120)
    raw = Path(config["data"]["tushare_raw_dir"])
    membership = weights.loc[
        weights["trade_date"].between(oos_start, oos_end),
        ["trade_date", "instrument"],
    ].copy()
    instruments = sorted(membership["instrument"].unique())
    prices = qlib_features(
        access,
        instruments,
        history_start,
        oos_end,
        float(config["tradability"]["qlib_amount_multiplier_to_cny"]),
    )
    lookback = int(config["tradability"]["amount_lookback_trade_days"])
    prices["avg_amount_cny"] = prices.groupby("instrument", sort=False)[
        "amount_cny"
    ].transform(
        lambda series: series.fillna(0).rolling(
            lookback, min_periods=lookback
        ).mean()
    )
    columns = [
        "trade_date",
        "instrument",
        "raw_high",
        "raw_low",
        "raw_close",
        "amount_cny",
        "avg_amount_cny",
    ]
    tradability = membership.merge(
        prices[columns],
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    limits = load_d2_yearly(
        access,
        config,
        "stk_limit",
        "trade_date",
        oos_start.year,
        oos_end.year,
    )
    suspensions = load_d2_yearly(
        access,
        config,
        "suspend_d",
        "trade_date",
        oos_start.year,
        oos_end.year,
    )
    limit_rows = limits.assign(
        instrument=limits["ts_code"].map(to_qlib_symbol)
    ).drop_duplicates(["trade_date", "instrument"])
    tradability = tradability.merge(
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
        .assign(suspend_record=True)
    )
    tradability = tradability.merge(
        suspended,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    stock_basic = access.read_parquet(
        raw / "stock_basic.parquet", date_col="list_date", end_date=oos_end
    )
    stock_basic["list_date"] = parse_date_series(
        stock_basic["list_date"], errors="coerce"
    )
    listings = (
        stock_basic.assign(instrument=stock_basic["ts_code"].map(to_qlib_symbol))
        .groupby("instrument", as_index=False)["list_date"]
        .min()
    )
    tradability = tradability.merge(
        listings, on="instrument", how="left", validate="many_to_one"
    )
    if tradability["list_date"].isna().any():
        raise ValueError("D9 部分成分股缺少上市日期")
    oos_calendar = calendar[(calendar >= oos_start) & (calendar <= oos_end)]
    tradability["listing_trade_days"] = trade_day_age(
        tradability["trade_date"], tradability["list_date"], calendar
    )
    namechange = access.read_parquet(
        raw / "namechange.parquet", date_col="start_date", end_date=oos_end
    )
    st_pairs = make_st_pairs(namechange, oos_calendar, oos_start, oos_end)
    tradability = tradability.merge(
        st_pairs[["trade_date", "instrument", "is_st"]].drop_duplicates(),
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    tradability["is_st"] = tradability["is_st"].eq(True)
    tradability["is_suspended"] = (
        tradability["suspend_record"].eq(True)
        | tradability["raw_close"].isna()
    )
    rtol = float(config["tradability"]["one_price_relative_tolerance"])
    atol = float(config["tradability"]["one_price_absolute_tolerance_cny"])
    one_price = is_close(
        tradability["raw_high"], tradability["raw_low"], rtol, atol
    )
    tradability["is_one_price_limit_up"] = (
        one_price
        & is_close(tradability["raw_close"], tradability["up_limit"], rtol, atol)
    ).fillna(False)
    tradability["is_one_price_limit_down"] = (
        one_price
        & is_close(tradability["raw_close"], tradability["down_limit"], rtol, atol)
    ).fillna(False)
    tradability["is_listed_under_120d"] = tradability[
        "listing_trade_days"
    ].lt(int(config["tradability"]["listing_min_trade_days"]))
    tradability["is_below_amount_floor"] = (
        tradability["avg_amount_cny"].isna()
        | tradability["avg_amount_cny"].lt(
            float(config["tradability"]["min_avg_daily_amount_cny"])
        )
    )
    tradability["is_eligible"] = ~(
        tradability["is_st"]
        | tradability["is_listed_under_120d"]
        | tradability["is_below_amount_floor"]
    )
    tradability["can_buy"] = (
        tradability["is_eligible"]
        & ~tradability["is_suspended"]
        & ~tradability["is_one_price_limit_up"]
    )
    tradability["can_sell"] = (
        ~tradability["is_suspended"]
        & ~tradability["is_one_price_limit_down"]
    )
    tradability = tradability.drop(columns=["suspend_record"])
    if tradability.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("D9 可交易性标记存在重复键")
    return tradability, {
        "start": str(tradability["trade_date"].min().date()),
        "end": str(tradability["trade_date"].max().date()),
        "rows": int(len(tradability)),
        "can_buy_rate": float(tradability["can_buy"].mean()),
        "can_sell_rate": float(tradability["can_sell"].mean()),
        "suspended_rows": int(tradability["is_suspended"].sum()),
        "st_rows": int(tradability["is_st"].sum()),
        "young_rows": int(tradability["is_listed_under_120d"].sum()),
        "one_price_up_rows": int(tradability["is_one_price_limit_up"].sum()),
        "one_price_down_rows": int(tradability["is_one_price_limit_down"].sum()),
    }


def load_weekly_alpha158(
    access: ResearchDataAccess,
    instruments: list[str],
    sample_dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = int(config["d4"]["expected_feature_count"])
    frames: list[pd.DataFrame] = []
    reference_spec: pd.DataFrame | None = None
    for year in range(sample_dates.min().year, sample_dates.max().year + 1):
        dates = sample_dates[sample_dates.year == year]
        if dates.empty:
            continue
        features, spec = load_alpha158_features(
            access, instruments, dates.min(), dates.max(), expected
        )
        features = features.loc[features["sample_date"].isin(dates)].copy()
        frames.append(features)
        if reference_spec is None:
            reference_spec = spec
        else:
            pd.testing.assert_frame_equal(reference_spec, spec)
        print(
            f"D9 Alpha158 {year}：{features['sample_date'].nunique()} 个周截面",
            flush=True,
        )
    if not frames or reference_spec is None:
        raise ValueError("D9 没有加载到 Alpha158 周截面")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("D9 Alpha158 周截面存在重复键")
    return result, reference_spec


def load_vwap_by_year(
    access: ResearchDataAccess,
    instruments: list[str],
    start,
    end,
) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    frames = []
    for year in range(start.year, end.year + 1):
        part_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        part_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        frames.append(load_vwap(access, instruments, part_start, part_end))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("D9 VWAP 存在重复键")
    return result


def build_model_dataset(
    access: ResearchDataAccess,
    config: dict,
    weights: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    design_start = pd.Timestamp(config["windows"]["design_start"])
    oos_end = pd.Timestamp(config["windows"]["oos_end"])
    weights = weights.loc[weights["trade_date"].between(design_start, oos_end)]
    sample_dates = select_weekly_sample_dates(weights)
    instruments = sorted(weights["instrument"].astype(str).unique())
    features, feature_spec = load_weekly_alpha158(
        access, instruments, sample_dates, config
    )
    vwap = load_vwap_by_year(
        access, instruments, design_start, oos_end
    )
    horizon = int(config["label"]["horizon_trade_days"])
    labels, coverage = build_forward_excess_labels(
        vwap, weights, sample_dates, calendar, [horizon]
    )
    dataset = labels.merge(
        features,
        on=["sample_date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    feature_names = feature_spec["feature"].tolist()
    dataset[feature_names] = dataset[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )
    if dataset.empty:
        raise ValueError("D9 Alpha158 与标签没有交集")
    if dataset["sample_date"].max() >= pd.Timestamp("2026-09-01"):
        raise OOSIntegrityError("D9 模型数据越过预注册样本外终点")
    return dataset, feature_names, feature_spec, coverage


def run_walk_forward_models(
    dataset: pd.DataFrame,
    feature_names: list[str],
    config: dict,
    *,
    artifact_prefix: str = "d9",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    windows = config["windows"]
    d4 = config["d4"]
    d5 = config["d5"]
    reports = ROOT / config["data"]["reports_dir"]
    models = ROOT / config["data"]["models_dir"]
    folds = build_expanding_year_folds(
        train_start=d4["train_start"],
        initial_train_end=windows["design_end"],
        prediction_start=windows["oos_start"],
        prediction_end=windows["oos_end"],
    )
    fold_records = pd.DataFrame([fold.as_record() for fold in folds])
    fold_validation = validate_oos_fold_records(
        fold_records,
        oos_start=windows["oos_start"],
        oos_end=windows["oos_end"],
    )
    if len(folds) != int(config["d9"]["expected_fold_count"]):
        raise OOSIntegrityError("D9 折数与预注册 expected_fold_count 不一致")
    design_history = pd.read_parquet(reports / "d5_walk_forward_predictions.parquet")
    design_history["sample_date"] = parse_date_series(design_history["sample_date"])
    history = design_history[["sample_date", "instrument", "score_z", "excess_return"]].copy()
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    model_parameters: dict | None = None
    lightgbm_version: str | None = None
    expected_column = str(d5["expected_return_column"])
    seed = int(config["project"]["random_seed"])
    first_expected = float(
        json.loads((models / "d5_calibration.json").read_text(encoding="utf-8"))[
            "coefficient"
        ]
    )
    for fold in folds:
        train, prediction = split_walk_forward_samples(dataset, fold)
        calibration = fit_return_calibration(history)
        if pd.Timestamp(calibration["history_end"]) >= fold.prediction_start:
            raise OOSIntegrityError(
                f"{fold.prediction_year} 校准历史越过预测起点"
            )
        if fold.prediction_year == folds[0].prediction_year and not np.isclose(
            float(calibration["coefficient"]), first_expected, rtol=1e-12, atol=1e-15
        ):
            raise OOSIntegrityError("2019 校准系数未复用冻结 D5 设计期结果")
        model_path = models / f"{artifact_prefix}_lightgbm_{fold.prediction_year}.txt"
        _, predicted, importance, params, version = fit_default_lightgbm(
            train, prediction, feature_names, seed, model_path
        )
        predicted["score_z"] = rank_gaussianize(predicted)
        if predicted["score_z"].isna().any():
            raise ValueError(f"{fold.prediction_year} 折存在无法正态化的分数")
        predicted[expected_column] = apply_return_calibration(
            predicted["score_z"], float(calibration["coefficient"])
        )
        predicted["prediction_year"] = fold.prediction_year
        predicted["model_train_end"] = fold.train_end
        prediction_frames.append(predicted)
        history = pd.concat(
            [
                history,
                predicted[["sample_date", "instrument", "score_z", "excess_return"]],
            ],
            ignore_index=True,
        )
        ic = summarize_ic(rank_ic_series(predicted, min_cross_section_size=2)["rank_ic"])
        attempted = pd.DatetimeIndex(
            dataset.loc[
                dataset["sample_date"].between(fold.train_start, fold.train_end),
                "sample_date",
            ].unique()
        )
        actual = pd.DatetimeIndex(train["sample_date"].unique())
        audit_rows.append(
            {
                **fold.as_record(),
                "train_rows": int(len(train)),
                "train_dates": int(train["sample_date"].nunique()),
                "prediction_rows": int(len(predicted)),
                "prediction_dates": int(predicted["sample_date"].nunique()),
                "last_train_sample_date": str(train["sample_date"].max().date()),
                "last_train_label_end": str(train["label_end_timestamp"].max().date()),
                "first_prediction_sample_date": str(predicted["sample_date"].min().date()),
                "last_prediction_sample_date": str(predicted["sample_date"].max().date()),
                "purged_train_dates": int(len(attempted.difference(actual))),
                "purge_passed": bool(
                    train["label_end_timestamp"].max() <= fold.train_end
                ),
                "rank_ic_mean": ic["rank_ic_mean"],
                "icir": ic["icir"],
                "calibration_coefficient": float(calibration["coefficient"]),
                "calibration_history_end": calibration["history_end"],
                "model_file": str(model_path.relative_to(ROOT)),
            }
        )
        calibration_rows.append(
            {
                "effective_prediction_year": fold.prediction_year,
                "calibration_coefficient": float(calibration["coefficient"]),
                "history_start": calibration["history_start"],
                "history_end": calibration["history_end"],
                "observations": int(calibration["observations"]),
                "dates": int(calibration["dates"]),
                "history_precedes_effective_year": True,
            }
        )
        importance = importance.copy()
        importance["prediction_year"] = fold.prediction_year
        importance_frames.append(importance)
        model_parameters = params
        lightgbm_version = version
        print(
            f"{artifact_prefix.upper()} 第 {fold.prediction_year} 折完成："
            f"train_end={fold.train_end.date()}，Rank IC={ic['rank_ic_mean']:.4f}",
            flush=True,
        )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["sample_date", "instrument"]
    )
    audits = pd.DataFrame(audit_rows).sort_values("prediction_year")
    calibrations = pd.DataFrame(calibration_rows).sort_values(
        "effective_prediction_year"
    )
    importance = pd.concat(importance_frames, ignore_index=True)
    return predictions, audits, calibrations, importance, {
        "fold_validation": fold_validation,
        "lightgbm_version": lightgbm_version,
        "model_parameters": model_parameters,
    }


def build_oos_exposures(
    access: ResearchDataAccess,
    config: dict,
    weights: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], pd.DataFrame]:
    d6 = config["d6"]
    oos_start = pd.Timestamp(config["windows"]["oos_start"])
    oos_end = pd.Timestamp(config["windows"]["oos_end"])
    history_start = oos_start - pd.Timedelta(
        days=int(d6["history_buffer_calendar_days"])
    )
    oos_weights = weights.loc[weights["trade_date"].between(oos_start, oos_end)]
    instruments = sorted(oos_weights["instrument"].astype(str).unique())
    mapping, catalog = require_industry_files(config)
    require_industry_coverage(instruments, mapping)
    if tuple(d6["style_factors"]) != STYLE_FACTORS:
        raise OOSIntegrityError("D9 检测到冻结 D6 风格因子列表变化")
    close = load_close_prices(access, instruments, history_start, oos_end)
    benchmark = load_benchmark_returns(
        access, config, history_start, oos_end
    )
    daily_basic = load_daily_basic(
        access, config, instruments, history_start, oos_end
    )
    raw_styles = compute_raw_style_history(
        close,
        benchmark,
        daily_basic,
        beta_window=int(d6["beta_window_trade_days"]),
        beta_minimum_observations=int(d6["beta_minimum_observations"]),
        momentum_window=int(d6["momentum_window_trade_days"]),
        momentum_skip=int(d6["momentum_skip_trade_days"]),
        liquidity_window=int(d6["liquidity_window_trade_days"]),
        liquidity_minimum_observations=int(d6["liquidity_minimum_observations"]),
    )
    eligible = usable_style_dates(raw_styles, oos_start, oos_end)
    eligible = eligible.intersection(signal_dates)
    missing = signal_dates.difference(eligible)
    if len(missing):
        raise ValueError(
            "D9 预测日缺少可用风格暴露："
            + ", ".join(str(value.date()) for value in missing[:20])
        )
    exposures, style_audit, industries, styles = assemble_exposures(
        oos_weights,
        raw_styles,
        mapping,
        catalog,
        eligible,
        winsor_mad_multiple=float(d6["winsor_mad_multiple"]),
    )
    matrix_audit, benchmark_exposures = audit_weekly_matrices(
        exposures,
        signal_dates,
        industries,
        styles,
        normalization_tolerance=float(d6["normalization_tolerance"]),
        industry_exposure_tolerance=float(d6["industry_exposure_tolerance"]),
    )
    if not matrix_audit["passed"].all():
        raise ValueError("D9 样本外因子暴露矩阵审计未通过")
    return exposures, style_audit, matrix_audit, industries, styles, close


def prepare_oos_market(
    access: ResearchDataAccess,
    config: dict,
    weights: pd.DataFrame,
    tradability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d6 = config["d6"]
    oos_start = pd.Timestamp(config["windows"]["oos_start"])
    oos_end = pd.Timestamp(config["windows"]["oos_end"])
    history_start = oos_start - pd.Timedelta(
        days=int(d6["history_buffer_calendar_days"])
    )
    instruments = sorted(
        weights.loc[
            weights["trade_date"].between(oos_start, oos_end), "instrument"
        ].unique()
    )
    market = load_qlib_market(
        access,
        instruments,
        history_start,
        oos_end,
        float(config["tradability"]["qlib_amount_multiplier_to_cny"]),
    )
    limits = load_d3_yearly(
        access, config, "stk_limit", oos_start.year, oos_end.year
    )
    suspensions = load_d3_yearly(
        access, config, "suspend_d", oos_start.year, oos_end.year
    )
    market = add_exchange_flags(market, limits, suspensions, config)
    market = add_ex_ante_volatility(market, config)
    strategy, _ = prepare_markets(market, tradability)
    strategy = strategy.loc[
        strategy["trade_date"].between(oos_start, oos_end)
    ].copy()
    return market, strategy


def audit_v2_turnover_path(
    predictions: pd.DataFrame,
    exposures: pd.DataFrame,
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    industry_columns: list[str],
    config: dict,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Replay target construction only; never construct portfolio performance."""

    d7 = config["d7"]
    limits, style_columns = validate_d7_config(config)
    signal_dates = pd.DatetimeIndex(
        predictions["sample_date"].unique()
    ).normalize().sort_values()
    schedule = make_execution_schedule(signal_dates, calendar)
    execution_by_signal = schedule.set_index("signal_date")["execution_date"]
    price_matrix = build_execution_price_matrix(market)
    expected_column = str(config["d5"]["expected_return_column"])
    cost_model = CostModel.from_config(config)

    def inputs_for_date(signal_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        cross = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross["benchmark_weight"], errors="coerce")
        benchmark = benchmark / benchmark.sum()
        observed = predictions.loc[predictions["sample_date"].eq(signal_date)]
        unexpected = sorted(set(observed["instrument"]) - set(cross.index))
        if unexpected:
            raise ValueError(
                f"{signal_date.date()} D9 V2 审计含非当日成分："
                + ", ".join(unexpected[:20])
            )
        mu = pd.to_numeric(
            observed.set_index("instrument")[expected_column], errors="coerce"
        ).reindex(cross.index)
        if mu.loc[mu.index.isin(observed["instrument"])].isna().any():
            raise ValueError(f"{signal_date.date()} D9 V2 审计 μ 含非有限值")
        return cross, benchmark, mu.fillna(0.0).astype(float)

    frozen_failure: dict[str, object] | None = None
    frozen_previous: pd.Series | None = None
    frozen_execution: pd.Timestamp | None = None
    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross, benchmark, mu = inputs_for_date(signal_date)
        if frozen_previous is None:
            pretrade = benchmark.copy()
        else:
            held = frozen_previous[
                frozen_previous > limits.tolerance * 0.1
            ].index
            pretrade = drift_pretrade_weights(
                frozen_previous,
                holding_returns_between(
                    price_matrix, frozen_execution, execution_date, held
                ),
                weight_tolerance=1e-8,
            )
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
                cross,
                industry_columns,
                style_columns,
                cost_rate=cost_rate,
                limits=limits,
                solver=str(d7["solver"]),
            )
        except PortfolioOptimizationError as exc:
            feasibility = audit_minimum_turnover_feasibility(
                mu,
                benchmark,
                pretrade,
                cross,
                industry_columns,
                style_columns,
                limits=limits,
                solver=str(d7["solver"]),
            )
            frozen_failure = {
                "signal_number": number,
                "signal_date": str(signal_date.date()),
                "execution_date": str(execution_date.date()),
                "original_error_type": type(exc).__name__,
                "original_error": str(exc),
                **feasibility,
            }
            break
        frozen_previous = result.weights
        frozen_execution = execution_date
    if frozen_failure is None:
        raise OOSIntegrityError("D9 V2 审计未复现已登记的冻结优化不可行")
    if frozen_failure["frozen_rule_feasible"] is not False:
        raise OOSIntegrityError("冻结优化失败但最小换手审计未证明 15% 上限冲突")
    if frozen_failure["v2_rule_feasible"] is not True:
        raise OOSIntegrityError("D9 V2 换手修订不能恢复原失败日可行性")

    v2_rows: list[dict[str, object]] = []
    v2_previous: pd.Series | None = None
    v2_execution: pd.Timestamp | None = None
    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross, benchmark, mu = inputs_for_date(signal_date)
        if v2_previous is None:
            pretrade = benchmark.copy()
        else:
            held = v2_previous[v2_previous > limits.tolerance * 0.1].index
            pretrade = drift_pretrade_weights(
                v2_previous,
                holding_returns_between(
                    price_matrix, v2_execution, execution_date, held
                ),
                weight_tolerance=1e-8,
            )
        cost_rate = symmetric_l1_cost_rate(
            cost_model.commission_rate,
            cost_model.transfer_fee_rate,
            cost_model.stamp_rate(execution_date),
        )
        result = optimize_long_only_active_portfolio_v2(
            mu,
            benchmark,
            pretrade,
            cross,
            industry_columns,
            style_columns,
            cost_rate=cost_rate,
            limits=limits,
            solver=str(d7["solver"]),
        )
        audit = result.audit
        v2_rows.append(
            {
                "phase": "v2_full_path",
                "signal_number": number,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "solver_status": result.status,
                "total_single_sided_turnover": audit["single_sided_turnover"],
                "mandatory_membership_turnover": audit[
                    "mandatory_membership_turnover"
                ],
                "discretionary_single_sided_turnover": audit[
                    "discretionary_single_sided_turnover"
                ],
                "v2_total_turnover_budget": audit["v2_total_turnover_budget"],
                "maximum_stock_active": audit["maximum_stock_active"],
                "maximum_industry_active": audit["maximum_industry_active"],
                "maximum_style_active": audit["maximum_style_active"],
                "all_constraints_passed": audit["passed"],
                "is_frozen_first_failure_date": bool(
                    str(signal_date.date()) == frozen_failure["signal_date"]
                ),
            }
        )
        v2_previous = result.weights
        v2_execution = execution_date
        if number == 1 or number % 40 == 0 or number == len(signal_dates):
            print(
                f"D9 V2 可行性审计 {number}/{len(signal_dates)}："
                f"{signal_date.date()}",
                flush=True,
            )
    dates = pd.DataFrame(v2_rows)
    failure_row = {
        "phase": "frozen_first_failure_minimum_turnover",
        **{
            key: value for key, value in frozen_failure.items()
            if not isinstance(value, (dict, list, tuple, set))
        },
    }
    failure_row["signal_date"] = pd.Timestamp(frozen_failure["signal_date"])
    failure_row["execution_date"] = pd.Timestamp(
        frozen_failure["execution_date"]
    )
    dates = pd.concat([dates, pd.DataFrame([failure_row])], ignore_index=True)
    payload = {
        "status": "passed",
        "classification": "post_oos_methodology_amendment",
        "audit_scope": "target_optimization_feasibility_only",
        "frozen_first_failure": frozen_failure,
        "v2_all_dates_feasible": bool(
            len(v2_rows) == len(signal_dates)
            and pd.DataFrame(v2_rows)["all_constraints_passed"].all()
        ),
        "v2_signal_dates": int(len(v2_rows)),
        "v2_maximum_total_single_sided_turnover": float(
            pd.DataFrame(v2_rows)["total_single_sided_turnover"].max()
        ),
        "v2_maximum_mandatory_membership_turnover": float(
            pd.DataFrame(v2_rows)["mandatory_membership_turnover"].max()
        ),
        "v2_maximum_discretionary_single_sided_turnover": float(
            pd.DataFrame(v2_rows)["discretionary_single_sided_turnover"].max()
        ),
        "v2_fallback_count": 0,
        "portfolio_backtest_run": False,
        "portfolio_performance_generated": False,
        "underlying_prices_read_for_causal_pretrade_drift": True,
        "rank_ic_was_already_observed_in_failed_recovery": True,
        "methodology_change": (
            "15% 上限由总单边换手改为主动单边换手；强制成分退出单独列示，"
            "仍计入总成交、目标函数线性成本和后续真实回测成本"
        ),
        "unchanged": [
            "D1-D8 frozen files and configuration",
            "alpha model and hyperparameters",
            "stock, industry and style constraints",
            "execution, tradability and transaction-cost rules",
            "OOS window and reporting thresholds",
        ],
    }
    return payload, dates


def generate_v2_layer5_targets(
    predictions: pd.DataFrame,
    exposures: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    execution_by_signal: pd.Series,
    pretrade_by_signal: dict[pd.Timestamp, pd.Series],
    industry_columns: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild only loss-ladder layer 5 with the V2 turnover definition."""

    limits, styles = optimizer_limits_for_layer(config, 5)
    expected_column = str(config["d5"]["expected_return_column"])
    cost_model = CostModel.from_config(config)
    target_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for signal_date in signal_dates:
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross["benchmark_weight"], errors="coerce")
        benchmark = benchmark / benchmark.sum()
        observed = predictions.loc[predictions["sample_date"].eq(signal_date)]
        unexpected = sorted(set(observed["instrument"]) - set(cross.index))
        if unexpected:
            raise ValueError(
                f"{signal_date.date()} D9 V2 第5层含非当日成分："
                + ", ".join(unexpected[:20])
            )
        mu = pd.to_numeric(
            observed.set_index("instrument")[expected_column],
            errors="coerce",
        ).reindex(cross.index)
        if mu.loc[mu.index.isin(observed["instrument"])].isna().any():
            raise ValueError(f"{signal_date.date()} D9 V2 第5层 μ 含非有限值")
        mu = mu.fillna(0.0)
        cost_rate = symmetric_l1_cost_rate(
            cost_model.commission_rate,
            cost_model.transfer_fee_rate,
            cost_model.stamp_rate(execution_date),
        )
        result = optimize_long_only_active_portfolio_v2(
            mu,
            benchmark,
            pretrade_by_signal[signal_date],
            cross,
            industry_columns,
            styles,
            cost_rate=cost_rate,
            limits=limits,
            solver=str(config["d7"]["solver"]),
        )
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
        audit_rows.append(
            {
                "layer": 5,
                "signal_date": signal_date,
                "solver_status": result.status,
                "fallback_used": False,
                "objective_value": result.objective_value,
                "estimated_linear_cost": result.estimated_linear_cost,
                "optimizer_single_sided_turnover": result.audit[
                    "single_sided_turnover"
                ],
                "optimizer_mandatory_membership_turnover": result.audit[
                    "mandatory_membership_turnover"
                ],
                "optimizer_discretionary_single_sided_turnover": result.audit[
                    "discretionary_single_sided_turnover"
                ],
            }
        )
    return pd.concat(target_frames, ignore_index=True), pd.DataFrame(audit_rows)


def generate_final_targets(
    predictions: pd.DataFrame,
    exposures: pd.DataFrame,
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    industry_columns: list[str],
    config: dict,
    *,
    turnover_policy: str = "frozen_total",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    if turnover_policy not in {"frozen_total", "v2_discretionary"}:
        raise ValueError(f"未知换手约束口径：{turnover_policy}")
    d6 = config["d6"]
    d7 = config["d7"]
    limits, style_columns = validate_d7_config(config)
    signal_dates = pd.DatetimeIndex(
        predictions["sample_date"].unique()
    ).normalize().sort_values()
    schedule = make_execution_schedule(signal_dates, calendar)
    execution_by_signal = schedule.set_index("signal_date")["execution_date"]
    price_matrix = build_execution_price_matrix(market)
    stock_returns = build_stock_returns(market)
    cost_model = CostModel.from_config(config)
    expected_column = str(config["d5"]["expected_return_column"])
    target_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    covariance_rows: list[dict[str, object]] = []
    previous_target: pd.Series | None = None
    previous_execution: pd.Timestamp | None = None
    for number, signal_date in enumerate(signal_dates, start=1):
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = pd.Timestamp(execution_by_signal.loc[signal_date]).normalize()
        cross = (
            exposures.loc[exposures["trade_date"].eq(signal_date)]
            .set_index("instrument")
            .sort_index()
        )
        benchmark = pd.to_numeric(cross["benchmark_weight"], errors="coerce")
        benchmark = benchmark / benchmark.sum()
        observed = predictions.loc[predictions["sample_date"].eq(signal_date)]
        mu_observed = observed.set_index("instrument")[expected_column]
        unexpected = sorted(set(mu_observed.index) - set(cross.index))
        if unexpected:
            raise ValueError(
                f"{signal_date.date()} D9 预测含非当日成分："
                + ", ".join(unexpected[:20])
            )
        mu = pd.to_numeric(mu_observed, errors="coerce").reindex(cross.index)
        if mu.loc[mu.index.isin(mu_observed.index)].isna().any():
            raise ValueError(f"{signal_date.date()} D9 已输出的 μ 含非有限值")
        coverage = float(mu.notna().mean())
        mu = mu.fillna(0.0).astype(float)
        if previous_target is None:
            pretrade = benchmark.copy()
            pretrade_source = "benchmark_seed"
        else:
            held = previous_target[previous_target > limits.tolerance * 0.1].index
            returns = holding_returns_between(
                price_matrix, previous_execution, execution_date, held
            )
            pretrade = drift_pretrade_weights(
                previous_target, returns, weight_tolerance=1e-8
            )
            pretrade_source = "prior_target_drifted_execution_to_execution"
        cost_rate = symmetric_l1_cost_rate(
            cost_model.commission_rate,
            cost_model.transfer_fee_rate,
            cost_model.stamp_rate(execution_date),
        )
        optimizer = (
            optimize_long_only_active_portfolio_v2
            if turnover_policy == "v2_discretionary"
            else optimize_long_only_active_portfolio
        )
        result = optimizer(
            mu,
            benchmark,
            pretrade,
            cross,
            industry_columns,
            style_columns,
            cost_rate=cost_rate,
            limits=limits,
            solver=str(d7["solver"]),
        )
        active = result.weights - benchmark
        covariance = estimate_ledoit_wolf_covariance(
            stock_returns,
            as_of_date=signal_date,
            instruments=cross.index,
            window=int(d6["covariance_window_trade_days"]),
            minimum_observations=int(d6["covariance_minimum_observations"]),
        )
        te = annualized_tracking_error(
            active,
            covariance.covariance,
            covariance.instruments,
            annualization_factor=float(d7["covariance_annualization_factor"]),
        )
        target_frames.append(
            pd.DataFrame(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "instrument": cross.index,
                    "target_weight": result.weights,
                    "benchmark_weight": benchmark,
                    "active_weight": active,
                    "pretrade_weight": pretrade.reindex(cross.index, fill_value=0.0),
                    expected_column: mu,
                    "mu_observed": mu.index.isin(mu_observed.index),
                }
            ).reset_index(drop=True)
        )
        audit = result.audit
        audit_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "prediction_year": signal_date.year,
                "solver_status": result.status,
                "fallback_used": False,
                "pretrade_source": pretrade_source,
                "cost_rate_per_l1": cost_rate,
                "mu_coverage": coverage,
                "objective_value": result.objective_value,
                "expected_5d_active_return": result.expected_active_return,
                "estimated_linear_cost": result.estimated_linear_cost,
                "weight_sum_error": audit["weight_sum_error"],
                "minimum_weight": audit["minimum_weight"],
                "maximum_stock_active": audit["maximum_stock_active"],
                "maximum_industry_active": audit["maximum_industry_active"],
                "maximum_style_active": audit["maximum_style_active"],
                "single_sided_turnover": audit["single_sided_turnover"],
                "forced_exit_weight": audit["forced_exit_weight"],
                "mandatory_membership_turnover": audit.get(
                    "mandatory_membership_turnover",
                    audit["forced_exit_weight"],
                ),
                "discretionary_single_sided_turnover": audit.get(
                    "discretionary_single_sided_turnover",
                    max(
                        float(audit["single_sided_turnover"])
                        - float(audit["forced_exit_weight"]),
                        0.0,
                    ),
                ),
                "holdings": int((result.weights > limits.tolerance).sum()),
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
        previous_execution = execution_date
        if number == 1 or number % 20 == 0 or number == len(signal_dates):
            print(
                f"D9{' V2' if turnover_policy == 'v2_discretionary' else ''} "
                f"最终目标 {number}/{len(signal_dates)}：{signal_date.date()}，"
                f"总换手 {audit['single_sided_turnover']:.2%}，"
                f"主动换手 {audit_rows[-1]['discretionary_single_sided_turnover']:.2%}，"
                f"TE {te.annualized_tracking_error:.2%}",
                flush=True,
            )
    return (
        pd.concat(target_frames, ignore_index=True),
        pd.DataFrame(audit_rows),
        pd.DataFrame(factor_rows),
        pd.DataFrame(covariance_rows),
        execution_by_signal,
    )


def yearly_performance(
    daily: pd.DataFrame,
    target_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    target = target_audit.copy()
    target["signal_date"] = pd.to_datetime(target["signal_date"])
    for year, group in frame.groupby(frame["trade_date"].dt.year, sort=True):
        metrics, monthly = summarize_monthly_performance(
            group,
            newey_west_lags=[min(3, max(0, group["trade_date"].dt.to_period("M").nunique() - 2))],
        )
        selected = target.loc[target["signal_date"].dt.year.eq(year)]
        rows.append(
            {
                "prediction_year": int(year),
                "months": int(len(monthly)),
                "annualized_excess_return": metrics["annualized_excess_return"],
                "annualized_tracking_error": metrics["annualized_tracking_error"],
                "information_ratio": metrics["information_ratio"],
                "newey_west_t": metrics["newey_west_t_minimum"],
                "maximum_relative_drawdown": metrics["maximum_relative_drawdown"],
                "portfolio_return": metrics["portfolio_cumulative_return"],
                "benchmark_return": metrics["benchmark_cumulative_return"],
                "excess_return_difference": metrics[
                    "cumulative_excess_return_difference"
                ],
                "target_turnover_median": float(
                    selected["single_sided_turnover"].median()
                ),
                "target_discretionary_turnover_median": float(
                    selected["discretionary_single_sided_turnover"].median()
                ) if "discretionary_single_sided_turnover" in selected else None,
                "actual_turnover_mean": float(
                    group.loc[group["signal_date"].notna(), "single_sided_turnover"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary: dict, reports_dir: Path, *, variant: str = "original"
) -> None:
    is_v2 = variant == "v2"
    performance = summary["performance"]
    thresholds = summary["threshold_evaluation"]
    acceptance = summary["execution_acceptance"]
    recovery = summary["repository"].get("membership_recovery")
    output_prefix = "d9_v2" if is_v2 else "d9"
    run_disclosure = (
        "- 本报告属于 **D9-V2 事后方法修订**，不是原预注册 D9 的无条件验证。"
        "原恢复运行已观察八折 Rank IC，但尚未生成组合收益、IR 或 NW-t；"
        "原始与恢复失败清单均原样保留。"
        if is_v2
        else (
        "- 本次是受控工程恢复后的唯一一次统计性 OOS：原流程在读取收益、训练模型或生成绩效前，"
        "因 35 个权重快照日的成分元数据不一致而停止；原失败清单保持不变，恢复清单另行留存。"
        if recovery
        else "- 本次是 2019-01 至 2026-08 样本外的唯一一次正式运行；运行清单在读取 OOS 前落盘。"
        )
    )
    lines = [
        "# D9-V2 滚动样本外报告" if is_v2 else "# D9 滚动样本外报告",
        "",
        f"**工程验收：{'通过' if acceptance['passed'] else '失败'}**",
        (
            f"**V2 事后参考判定：{thresholds['decision']}**"
            if is_v2 else f"**当前研究判定：{thresholds['decision']}**"
        ),
        "",
        "## 冻结与一次性运行",
        "",
        f"- 冻结提交：`{summary['repository']['frozen_oos_commit']}`；运行分支：`{summary['repository']['branch']}`。",
        "- D1–D8 的代码与配置值均未改变；仅新增 D9 文件与 config.yaml 的 d9 节。",
        run_disclosure,
        (
            "- V2 唯一口径修订：15% 约束主动单边换手；强制成分退出单独列示，"
            "但仍计入总成交、目标函数线性成本与真实回测交易成本。"
            if is_v2 else ""
        ),
        (
            "- D15 必须并列披露：原始元数据失败、恢复后的优化不可行失败、"
            "V2 可行性审计及本报告；V2 结论只能作为事后修订证据。"
            if is_v2 else ""
        ),
        (
            "- V2 继承已登记的成分元数据恢复；该恢复只修正指数事件与历史证券代码映射。"
            if is_v2
            else (
                "- 恢复只修正中证500成分事件与历史证券代码映射；未读取既有 OOS 绩效、未改模型、参数、门槛或交易口径。"
                if recovery else "- 未发生工程恢复。"
            )
        ),
        "- 冻结 D7 的事前 TE 中位数为 "
        f"{summary['frozen_d7_warning']['tracking_error_median']:.2%}，低于预注册 4% 下限；该失败原样保留，没有调参。",
        "",
        "## 固定口径",
        "",
        "- 8 折按自然年扩展窗口重训；Alpha158 + LightGBM 默认超参不变。",
        "- 每年校准系数只使用该年以前已实现的走向前预测；训练集末尾标签跨界样本全部 purge。",
        "- 周频信号、T+1 全天 VWAP、PIT 中证500成分、月度权重自然漂移、D3 动态成本与可交易性规则全部复用冻结实现。",
        "- IR 与 t 值只由日收益先复合成月度收益后计算；NW-t 同时报 lag 3–6，并按其中最小值判定。",
        (
            "- V2 仍使用冻结的 15% 数值上限；总单边换手 = 强制调样换手 + 主动单边换手。"
            if is_v2 else ""
        ),
        "",
        "## 八折结果",
        "",
        "| 年份 | 训练截止 | Rank IC | 费后年化超额 | 月频IR | NW-t | 最大相对回撤 | 目标/实际换手 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["yearly"]:
        lines.append(
            f"| {row['prediction_year']} | {row['train_end']} | {row['rank_ic_mean']:.4f} "
            f"| {row['annualized_excess_return']:.2%} | {row['information_ratio']:.3f} "
            f"| {row['newey_west_t']:.3f} | {row['maximum_relative_drawdown']:.2%} "
            f"| {row['target_turnover_median']:.2%}/{row['actual_turnover_mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 全期月频统计",
            "",
            f"- 费后年化超额：{performance['annualized_excess_return']:.2%}。",
            f"- 月频年化 TE：{performance['annualized_tracking_error']:.2%}；IR：{performance['information_ratio']:.3f}。",
            f"- 自洽检查 `IR × TE`：{performance['identity_ir_times_te']:.2%}。",
            f"- Newey-West t（lag 3–6）：{performance['newey_west_t_by_lag']}；保守判定值 {performance['newey_west_t_minimum']:.3f}。",
            f"- 最大相对回撤：{performance['maximum_relative_drawdown']:.2%}。",
            "",
            (
                "## 沿用原六项门槛（V2 事后参考）"
                if is_v2 else "## 六项预注册门槛"
            ),
            "",
            "| 门槛 | 实测 | 阈值 | 结果 |",
            "|---|---:|---:|:---:|",
            f"| 费后年化超额 | {thresholds['values']['annualized_excess_return']:.2%} | ≥3.5% | {'通过' if thresholds['checks']['annualized_excess_return'] else '未通过'} |",
            f"| 月频 IR | {thresholds['values']['information_ratio']:.3f} | ≥0.75 | {'通过' if thresholds['checks']['information_ratio'] else '未通过'} |",
            f"| Newey-West t | {thresholds['values']['newey_west_t']:.3f} | ≥2 | {'通过' if thresholds['checks']['newey_west_t'] else '未通过'} |",
            f"| 最大相对回撤 | {thresholds['values']['maximum_relative_drawdown']:.2%} | ≤10% | {'通过' if thresholds['checks']['maximum_relative_drawdown'] else '未通过'} |",
            "| 特异性贡献占比 | 待 D10 WLS 归因 | ≥50% | 待定 |",
            f"| 冲击系数×2后年化超额 | {thresholds['values']['doubled_impact_annualized_excess']:.2%} | >0 | {'通过' if thresholds['checks']['doubled_impact_positive'] else '未通过'} |",
            "",
            f"判定说明：{thresholds['reason']}。D10 完成特异性归因前，不允许把“待D10归因”升级为 Go。",
            "",
            "## 样本外损耗阶梯（主资金）",
            "",
            "| 层 | 设置 | 设计期年化超额 | 样本外年化超额 | 样本外目标换手 | 样本外实际换手 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    design = {int(row["layer"]): row for row in summary["design_ladder"]}
    for row in summary["oos_ladder"]:
        layer = int(row["layer"])
        lines.append(
            f"| {layer} | {LAYER_NAMES[layer]} | {design[layer]['annualized_excess_return']:.2%} "
            f"| {row['annualized_excess_return']:.2%} | {row['target_turnover_median']:.2%} "
            f"| {row['executed_turnover_mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            "第0–2层沿各自路径；第3–6层统一相对样本外最终策略的冻结事前持仓路径计算目标换手。目标换手与 D3 实际成交换手分列。",
            "",
            "## 输出",
            "",
            "- `reports/d9_run_manifest.json`、`reports/d9_recovery_manifest.json` "
            "及 V2 的独立 manifest（V2 时）",
            f"- `reports/{output_prefix}_membership_recovery_audit.json`、"
            f"`{output_prefix}_summary.json` 与版本化 Markdown 报告",
            f"- `reports/{output_prefix}_oos_predictions.parquet`、"
            f"`{output_prefix}_fold_audit.parquet`、"
            f"`{output_prefix}_calibration_schedule.parquet`",
            f"- `reports/{output_prefix}_target_weights.parquet`、"
            f"`{output_prefix}_constraint_audit.parquet`、"
            f"`{output_prefix}_covariance_audit.parquet`",
            f"- `reports/{output_prefix}_main_monthly.parquet`、"
            f"`{output_prefix}_yearly_performance.parquet`",
            f"- `reports/{output_prefix}_ladder_summary.parquet` 与每层 targets/daily/trades",
            f"- `reports/{output_prefix}_impact2x_daily.parquet`、"
            f"`{output_prefix}_impact2x_trades.parquet`",
            "",
        ]
    )
    path = reports_dir / (
        "D9_V2滚动样本外报告.md" if is_v2 else "D9滚动样本外报告.md"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def execute_v2_feasibility_audit(
    config: dict,
    repository_audit: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    access = ResearchDataAccess(ROOT)
    if access.mode != "oos":
        raise OOSIntegrityError("D9 V2 可行性审计要求 OOS 数据保持已解封状态")
    windows = config["windows"]
    oos_start = pd.Timestamp(windows["oos_start"])
    oos_end = pd.Timestamp(windows["oos_end"])
    calendar = load_open_calendar(access, config, end_date=oos_end)
    print("D9 V2 审计重建 PIT 权重与因果持仓路径……", flush=True)
    weights, weight_audit = build_full_benchmark_weights(access, config, calendar)
    tradability, _ = build_oos_tradability(access, config, weights, calendar)
    dataset, feature_names, _, _ = build_model_dataset(
        access, config, weights, calendar
    )
    predictions, fold_audit, _, _, _ = run_walk_forward_models(
        dataset,
        feature_names,
        config,
        artifact_prefix="d9_v2_feasibility",
    )
    signal_dates = pd.DatetimeIndex(
        predictions["sample_date"].unique()
    ).normalize().sort_values()
    exposures, _, matrix_audit, industries, _, _ = build_oos_exposures(
        access, config, weights, signal_dates
    )
    market, _ = prepare_oos_market(access, config, weights, tradability)
    oos_calendar = calendar[(calendar >= oos_start) & (calendar <= oos_end)]
    payload, dates = audit_v2_turnover_path(
        predictions,
        exposures,
        market,
        oos_calendar,
        industries,
        config,
    )
    payload.update(
        {
            "v2_head_commit": repository_audit["d9_head_commit"],
            "frozen_oos_commit": repository_audit["frozen_oos_commit"],
            "config_sha256": sha256(ROOT / "config/config.yaml"),
            "original_manifest_sha256": repository_audit[
                "v2_source_failures"
            ]["original_manifest_sha256"],
            "recovery_manifest_sha256": repository_audit[
                "v2_source_failures"
            ]["recovery_manifest_sha256"],
            "membership_metadata_mismatch_days": int(
                weight_audit["snapshot_mismatch_days"]
            ),
            "folds_replayed": int(len(fold_audit)),
            "factor_dates_passed": int(matrix_audit["passed"].sum()),
            "source_failure_validation": repository_audit[
                "v2_source_failures"
            ],
        }
    )
    return payload, dates


def execute_d9(
    config: dict,
    repository_audit: dict[str, object],
    upstream: dict[str, dict],
    *,
    variant: str = "original",
) -> dict:
    if variant not in {"original", "v2"}:
        raise ValueError(f"未知 D9 运行版本：{variant}")
    is_v2 = variant == "v2"
    artifact_prefix = "d9_v2" if is_v2 else "d9"
    access = ResearchDataAccess(ROOT)
    if access.mode != "oos":
        raise OOSIntegrityError(
            "正式 D9 运行前必须执行 seal_oos.py unseal --confirm UNSEAL_OOS"
        )
    reports = ROOT / config["data"]["reports_dir"]
    processed_oos = ROOT / (
        config["d9"]["v2"]["processed_oos_dir"]
        if is_v2
        else config["d9"]["processed_oos_dir"]
    )
    reports.mkdir(parents=True, exist_ok=True)
    processed_oos.mkdir(parents=True, exist_ok=True)
    windows = config["windows"]
    design_start = pd.Timestamp(windows["design_start"])
    oos_start = pd.Timestamp(windows["oos_start"])
    oos_end = pd.Timestamp(windows["oos_end"])
    calendar = load_open_calendar(access, config, end_date=oos_end)

    print("D9 构建完整 PIT 基准权重并核对冻结设计期……", flush=True)
    weights, weight_audit = build_full_benchmark_weights(
        access, config, calendar
    )
    atomic_json(
        weight_audit,
        reports / f"{artifact_prefix}_membership_recovery_audit.json",
    )
    tradability, tradability_audit = build_oos_tradability(
        access, config, weights, calendar
    )
    atomic_parquet(
        weights.loc[weights["trade_date"].between(oos_start, oos_end)],
        processed_oos / "benchmark_weights.parquet",
    )
    atomic_parquet(tradability, processed_oos / "tradability_flags.parquet")

    print("D9 分年读取 Alpha158 并生成固定 5 日标签……", flush=True)
    dataset, feature_names, feature_spec, label_coverage = build_model_dataset(
        access, config, weights, calendar
    )
    predictions, fold_audit, calibrations, importance, model_meta = (
        run_walk_forward_models(
            dataset,
            feature_names,
            config,
            artifact_prefix=artifact_prefix,
        )
    )
    signal_dates = pd.DatetimeIndex(
        predictions["sample_date"].unique()
    ).normalize().sort_values()

    print("D9 构建样本外因子暴露矩阵……", flush=True)
    exposures, style_audit, matrix_audit, industries, styles, _ = (
        build_oos_exposures(access, config, weights, signal_dates)
    )
    market, strategy_market = prepare_oos_market(
        access, config, weights, tradability
    )
    oos_calendar = calendar[(calendar >= oos_start) & (calendar <= oos_end)]
    targets, constraint_audit, factor_active, covariance_audit, execution_by_signal = (
        generate_final_targets(
            predictions,
            exposures,
            market,
            oos_calendar,
            industries,
            config,
            turnover_policy=(
                "v2_discretionary" if is_v2 else "frozen_total"
            ),
        )
    )

    price_matrix = build_execution_price_matrix(market)
    frozen_pretrade = build_frozen_d7_pretrade_path(
        targets,
        exposures,
        signal_dates,
        execution_by_signal,
        price_matrix,
        tolerance=float(config["d7"]["constraint_tolerance"]),
    )
    top = make_top_n_equal_targets(
        predictions,
        top_n=int(config["d8"]["top_n"]),
        score_col=str(config["d5"]["expected_return_column"]),
    )
    layer_targets: dict[int, pd.DataFrame] = {
        0: top.copy(),
        1: top.copy(),
        2: top.copy(),
        6: targets[["signal_date", "execution_date", "instrument", "target_weight"]].copy(),
    }
    solver_frames: list[pd.DataFrame] = []
    for layer in [3, 4, 5]:
        if is_v2 and layer == 5:
            layer_target, solver = generate_v2_layer5_targets(
                predictions,
                exposures,
                signal_dates,
                execution_by_signal,
                frozen_pretrade,
                industries,
                config,
            )
        else:
            layer_target, solver = generate_optimized_layer_targets(
                layer,
                predictions,
                exposures,
                signal_dates,
                execution_by_signal,
                frozen_pretrade,
                industries,
                config,
            )
        layer_targets[layer] = layer_target
        solver_frames.append(solver)
    target_audit, ladder_factor_active = audit_all_layer_targets(
        layer_targets,
        exposures,
        signal_dates,
        execution_by_signal,
        price_matrix,
        frozen_pretrade,
        industries,
        [f"style_{name}" for name in config["d7"]["constrained_style_factors"]],
        float(config["d7"]["constraint_tolerance"]),
    )
    target_audit["mandatory_membership_turnover"] = target_audit[
        "forced_exit_weight"
    ]
    target_audit["discretionary_single_sided_turnover"] = (
        target_audit["target_single_sided_turnover"]
        - target_audit["forced_exit_weight"]
    ).clip(lower=0.0)
    stock_returns = build_stock_returns(market)
    ex_ante_te = calculate_layer_tracking_error(
        layer_targets, exposures, stock_returns, signal_dates, config
    )
    fallback_amount = float(config["tradability"]["min_avg_daily_amount_cny"])
    ideal_market = make_ideal_execution_market(
        strategy_market, fallback_daily_amount=fallback_amount
    )
    benchmark = access.read_parquet(
        Path(config["data"]["tushare_raw_dir"])
        / "index_daily_H00905_CSI.parquet",
        date_col="trade_date",
        start_date=oos_start,
        end_date=oos_end,
    )
    benchmark["trade_date"] = parse_date_series(benchmark["trade_date"])
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    ladder, daily_by_layer, trades_by_layer = run_loss_ladder_backtests(
        layer_targets,
        ideal_market.copy(),
        ideal_market,
        strategy_market,
        benchmark,
        oos_calendar,
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
            maximum_discretionary_turnover=(
                "discretionary_single_sided_turnover", "max"
            ),
            discretionary_turnover_median=(
                "discretionary_single_sided_turnover", "median"
            ),
        )
        .reset_index()
    )
    te_summary = (
        ex_ante_te.groupby("layer")
        .agg(
            ex_ante_te_median=("annualized_ex_ante_tracking_error", "median"),
            covariance_active_l1_coverage_min=("covariance_active_l1_coverage", "min"),
        )
        .reset_index()
    )
    ladder = ladder.merge(target_summary, on="layer", how="left", validate="many_to_one")
    ladder = ladder.merge(te_summary, on="layer", how="left", validate="many_to_one")
    main_capital = float(config["d9"]["main_capital_cny"])
    if main_capital not in [float(value) for value in config["portfolio"]["capital_scenarios"]]:
        raise OOSIntegrityError("D9 主资金规模不在冻结 portfolio.capital_scenarios 中")
    main_daily = daily_by_layer[6].loc[
        np.isclose(daily_by_layer[6]["capital"], main_capital)
    ].copy()
    main_trades = trades_by_layer[6].loc[
        np.isclose(trades_by_layer[6]["capital"], main_capital)
    ].copy()
    performance, monthly = summarize_monthly_performance(
        main_daily,
        newey_west_lags=config["d9"]["newey_west_lags"],
    )
    yearly = yearly_performance(main_daily, constraint_audit)

    base_cost = CostModel.from_config(config)
    impact_multiplier = float(config["d9"]["impact_stress_multiplier"])
    stress_model = CostModel(
        commission_rate=base_cost.commission_rate,
        transfer_fee_rate=base_cost.transfer_fee_rate,
        stamp_duty=base_cost.stamp_duty,
        impact_coefficient=base_cost.impact_coefficient * impact_multiplier,
        participation_limit=base_cost.participation_limit,
    )
    stress_result = run_vwap_backtest(
        strategy_market,
        targets[["signal_date", "instrument", "target_weight"]],
        main_capital,
        stress_model,
    )
    _, stress_daily = summarize_backtest_performance(
        stress_result.nav,
        benchmark,
        initial_capital=main_capital,
        annualization_factor=float(config["d8"]["return_annualization_factor"]),
    )
    stress_performance, _ = summarize_monthly_performance(
        stress_daily,
        newey_west_lags=config["d9"]["newey_west_lags"],
    )
    thresholds = OOSThresholds.from_config(config["d9"]["thresholds"])
    threshold_evaluation = evaluate_oos_thresholds(
        performance,
        doubled_impact_annualized_excess=stress_performance[
            "annualized_excess_return"
        ],
        thresholds=thresholds,
        specific_contribution_share=None,
    )

    tolerance = float(config["d7"]["constraint_tolerance"])
    solver_audit = pd.concat(solver_frames, ignore_index=True)
    execution = summarize_execution(main_trades, oos_calendar)
    turnover_acceptance_column = (
        "maximum_discretionary_turnover" if is_v2
        else "maximum_target_turnover"
    )
    execution_acceptance = {
        "frozen_repository_verified": True,
        "one_shot_manifest_started_before_oos_read": True,
        "controlled_recovery_protocol_validated": bool(
            (
                is_v2
                and repository_audit.get("v2_source_failures", {}).get(
                    "validated"
                ) is True
            )
            or repository_audit.get("membership_recovery") is None
            or repository_audit["membership_recovery"].get("validated") is True
        ),
        "post_oos_amendment_disclosed": True,
        "eight_folds_complete": bool(
            len(fold_audit) == int(config["d9"]["expected_fold_count"])
        ),
        "purge_all_folds": bool(fold_audit["purge_passed"].all()),
        "calibration_history_prior_only": bool(
            calibrations["history_precedes_effective_year"].all()
        ),
        "final_solver_success_100pct": bool(
            constraint_audit["solver_status"].eq("optimal").all()
        ),
        "final_fallback_zero": bool(~constraint_audit["fallback_used"].any()),
        "final_constraints_passed": bool(
            constraint_audit["all_constraints_passed"].all()
        ),
        "covariance_coverage": bool(
            constraint_audit["covariance_active_l1_coverage"].min()
            >= float(config["d7"]["covariance_minimum_active_l1_coverage"])
        ),
        "ladder_solver_success_100pct": bool(
            solver_audit["solver_status"].eq("optimal").all()
        ),
        "ladder_fallback_zero": bool(~solver_audit["fallback_used"].any()),
        "industry_constraint": bool(
            target_summary.loc[target_summary["layer"].ge(3), "maximum_industry_active"].max()
            <= float(config["d7"]["industry_active_limit"]) + tolerance
        ),
        "style_constraint": bool(
            target_summary.loc[target_summary["layer"].ge(4), "maximum_style_active"].max()
            <= float(config["d7"]["style_active_limit"]) + tolerance
        ),
        "turnover_constraint": bool(
            target_summary.loc[
                target_summary["layer"].ge(5), turnover_acceptance_column
            ].max()
            <= float(config["d7"]["single_sided_turnover_limit"]) + tolerance
        ),
        "stock_constraint": bool(
            target_summary.loc[target_summary["layer"].eq(6), "maximum_stock_active"].iloc[0]
            <= float(config["d7"]["stock_active_limit"]) + tolerance
        ),
        "t_plus_one": bool(execution["t_plus_one_ok"]),
        "nw_monthly_frequency": True,
        "d10_attribution_not_fabricated": bool(
            threshold_evaluation["checks"]["specific_contribution_share"] is None
        ),
    }
    execution_acceptance["passed"] = bool(all(execution_acceptance.values()))
    fold_yearly = fold_audit.merge(yearly, on="prediction_year", validate="one_to_one")
    design_ladder = upstream["d8"]["layers"]
    main_ladder = ladder.loc[np.isclose(ladder["capital"], main_capital)].sort_values("layer")
    oos_ladder_json = [
        {
            "layer": int(row.layer),
            "annualized_excess_return": float(row.annualized_excess_return),
            "ex_ante_te_median": float(row.ex_ante_te_median),
            "target_turnover_median": float(row.target_turnover_median),
            "target_discretionary_turnover_median": float(
                row.discretionary_turnover_median
            ),
            "executed_turnover_mean": float(row.mean_rebalance_single_sided_turnover),
            "transaction_cost": float(row.total_transaction_cost),
            "impact_cost": float(row.impact_cost),
            "fill_ratio": float(row.fill_ratio),
            "blocked_by_tradability_rows": int(row.blocked_by_tradability_rows),
        }
        for row in main_ladder.itertuples(index=False)
    ]
    summary = {
        "version": "D9-V2" if is_v2 else "D9",
        "classification": (
            "post_oos_methodology_amendment" if is_v2
            else "preregistered_oos"
        ),
        "mode": access.mode,
        "oos_data_accessed": True,
        "oos_run_count": 1,
        "statistical_oos_run_count": 1,
        "post_oos_v2_run_count": 1 if is_v2 else 0,
        "engineering_recovery": repository_audit.get("membership_recovery"),
        "v2_source_failures": repository_audit.get("v2_source_failures"),
        "methodology_amendment": (
            {
                "rank_ic_observed_before_amendment": True,
                "portfolio_performance_observed_before_amendment": False,
                "reason": "冻结总换手 15% 与指数强制调样退出发生确定性可行域冲突",
                "changed_rule": (
                    "15% 上限约束主动单边换手；强制成分退出单独列示，"
                    "但仍计入总成交与全部交易成本"
                ),
                "numeric_limits_changed": False,
                "model_or_threshold_parameters_changed": False,
                "confirmatory_status": "不得表述为原预注册 D9 的无条件验证",
                "d15_reporting": "原两次失败与 V2 结果并列披露",
            }
            if is_v2
            else None
        ),
        "repository": repository_audit,
        "window": {
            "design_start": str(design_start.date()),
            "design_end": windows["design_end"],
            "oos_start": windows["oos_start"],
            "oos_end": windows["oos_end"],
        },
        "data": {
            "benchmark_weights": weight_audit,
            "tradability": tradability_audit,
            "label_minimum_weight_coverage": float(label_coverage["weight_coverage"].min()),
            "factor_matrix_dates": int(len(matrix_audit)),
            "factor_matrix_all_passed": bool(matrix_audit["passed"].all()),
        },
        "model": {
            "feature_set": "Alpha158",
            "feature_count": int(len(feature_spec)),
            "class": "lightgbm.LGBMRegressor",
            "parameter_policy": "D4 installed-version defaults unchanged; random_state fixed only",
            "lightgbm_version": model_meta["lightgbm_version"],
            "parameters": model_meta["model_parameters"],
            "fold_validation": model_meta["fold_validation"],
        },
        "solver": {
            "name": "CLARABEL",
            "optimal_dates": int(constraint_audit["solver_status"].eq("optimal").sum()),
            "total_dates": int(len(constraint_audit)),
            "fallback_count": int(constraint_audit["fallback_used"].sum()),
            "minimum_mu_coverage": float(constraint_audit["mu_coverage"].min()),
        },
        "constraints": {
            "maximum_stock_active": float(constraint_audit["maximum_stock_active"].max()),
            "maximum_industry_active": float(constraint_audit["maximum_industry_active"].max()),
            "maximum_style_active": float(constraint_audit["maximum_style_active"].max()),
            "maximum_target_single_sided_turnover": float(constraint_audit["single_sided_turnover"].max()),
            "maximum_mandatory_membership_turnover": float(
                constraint_audit["mandatory_membership_turnover"].max()
            ),
            "maximum_discretionary_single_sided_turnover": float(
                constraint_audit["discretionary_single_sided_turnover"].max()
            ),
            "turnover_acceptance_basis": (
                "discretionary_single_sided_turnover" if is_v2
                else "total_single_sided_turnover"
            ),
            "minimum_covariance_active_l1_coverage": float(constraint_audit["covariance_active_l1_coverage"].min()),
            "ex_ante_te_median": float(constraint_audit["annualized_tracking_error"].median()),
        },
        "performance": performance,
        "impact_2x": {
            "multiplier": impact_multiplier,
            "annualized_excess_return": stress_performance["annualized_excess_return"],
            "information_ratio": stress_performance["information_ratio"],
            "total_transaction_cost": float(stress_result.trades["total_cost"].sum()),
        },
        "threshold_evaluation": threshold_evaluation,
        "frozen_d7_warning": {
            "tracking_error_passed": bool(upstream["d7"]["acceptance"]["tracking_error"]),
            "tracking_error_median": float(upstream["d7"]["tracking_error"]["median"]),
            "tracking_error_acceptance_min": float(upstream["d7"]["tracking_error"]["acceptance_min"]),
            "parameters_changed": False,
            "turnover_definition_changed_in_v2": bool(is_v2),
        },
        "yearly": fold_yearly.to_dict(orient="records"),
        "design_ladder": design_ladder,
        "oos_ladder": oos_ladder_json,
        "execution": execution,
        "execution_acceptance": execution_acceptance,
    }

    keep_prediction = [
        "sample_date",
        "instrument",
        "prediction_year",
        "model_train_end",
        "prediction",
        "score_z",
        str(config["d5"]["expected_return_column"]),
        "excess_return",
        "stock_return",
        "benchmark_return",
        "label_start_timestamp",
        "label_end_timestamp",
        "weight_coverage",
    ]
    atomic_parquet(
        predictions[keep_prediction], reports / f"{artifact_prefix}_oos_predictions.parquet"
    )
    atomic_parquet(fold_audit, reports / f"{artifact_prefix}_fold_audit.parquet")
    atomic_parquet(
        calibrations, reports / f"{artifact_prefix}_calibration_schedule.parquet"
    )
    atomic_parquet(
        importance, reports / f"{artifact_prefix}_feature_importance.parquet"
    )
    atomic_parquet(feature_spec, reports / f"{artifact_prefix}_alpha158_spec.parquet")
    atomic_parquet(
        label_coverage, reports / f"{artifact_prefix}_label_coverage.parquet"
    )
    atomic_parquet(exposures, reports / f"{artifact_prefix}_factor_exposures.parquet")
    atomic_parquet(
        style_audit, reports / f"{artifact_prefix}_style_imputation.parquet"
    )
    atomic_parquet(matrix_audit, reports / f"{artifact_prefix}_factor_audit.parquet")
    atomic_parquet(targets, reports / f"{artifact_prefix}_target_weights.parquet")
    atomic_parquet(
        constraint_audit, reports / f"{artifact_prefix}_constraint_audit.parquet"
    )
    atomic_parquet(
        factor_active, reports / f"{artifact_prefix}_factor_active_exposures.parquet"
    )
    atomic_parquet(
        covariance_audit, reports / f"{artifact_prefix}_covariance_audit.parquet"
    )
    atomic_parquet(ladder, reports / f"{artifact_prefix}_ladder_summary.parquet")
    atomic_parquet(
        build_loss_deltas(ladder, capital=main_capital),
        reports / f"{artifact_prefix}_loss_deltas.parquet",
    )
    atomic_parquet(
        target_audit, reports / f"{artifact_prefix}_ladder_target_audit.parquet"
    )
    atomic_parquet(
        ladder_factor_active,
        reports / f"{artifact_prefix}_ladder_factor_active.parquet",
    )
    atomic_parquet(
        ex_ante_te, reports / f"{artifact_prefix}_ladder_ex_ante_te.parquet"
    )
    atomic_parquet(
        solver_audit, reports / f"{artifact_prefix}_ladder_solver_audit.parquet"
    )
    for layer in range(7):
        atomic_parquet(
            layer_targets[layer],
            reports / f"{artifact_prefix}_layer_{layer}_targets.parquet",
        )
        atomic_parquet(
            daily_by_layer[layer],
            reports / f"{artifact_prefix}_layer_{layer}_daily.parquet",
        )
        atomic_parquet(
            trades_by_layer[layer],
            reports / f"{artifact_prefix}_layer_{layer}_trades.parquet",
        )
    atomic_parquet(main_daily, reports / f"{artifact_prefix}_main_daily.parquet")
    atomic_parquet(monthly, reports / f"{artifact_prefix}_main_monthly.parquet")
    atomic_parquet(
        fold_yearly, reports / f"{artifact_prefix}_yearly_performance.parquet"
    )
    atomic_parquet(stress_daily, reports / f"{artifact_prefix}_impact2x_daily.parquet")
    atomic_parquet(
        stress_result.trades, reports / f"{artifact_prefix}_impact2x_trades.parquet"
    )
    atomic_json(summary, reports / f"{artifact_prefix}_summary.json")
    write_report(summary, reports, variant=variant)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D9 一次性滚动样本外回测"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="只检查冻结、分支、依赖与数据文件，不读取样本外收益",
    )
    mode.add_argument(
        "--recovery-preflight",
        action="store_true",
        help="核对已登记的 35 日成分失败，且不消耗受控恢复机会",
    )
    mode.add_argument(
        "--resume-membership-recovery",
        action="store_true",
        help="保留原始失败清单，以同一次统计性 OOS 做一次受控成分恢复",
    )
    mode.add_argument(
        "--v2-feasibility-audit",
        action="store_true",
        help="复现冻结优化失败并验证 V2 全日期可行性，不生成组合绩效",
    )
    mode.add_argument(
        "--run-v2",
        action="store_true",
        help="以独立清单运行已披露的 D9-V2 事后方法修订",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="受控恢复或 D9-V2 正式运行确认口令",
    )
    args = parser.parse_args()
    config = load_config()
    if args.confirm and not (args.resume_membership_recovery or args.run_v2):
        parser.error(
            "--confirm 只能与 --resume-membership-recovery 或 --run-v2 一起使用"
        )
    recovery_mode = bool(
        args.recovery_preflight or args.resume_membership_recovery
    )
    if args.resume_membership_recovery:
        expected_confirmation = str(
            config["d9"]["membership_recovery"]["confirmation"]
        )
        if args.confirm != expected_confirmation:
            raise OOSIntegrityError(
                "受控恢复必须显式传入 --confirm " + expected_confirmation
            )
    if args.run_v2:
        expected_confirmation = str(config["d9"]["v2"]["confirmation"])
        if args.confirm != expected_confirmation:
            raise OOSIntegrityError(
                "D9-V2 正式运行必须显式传入 --confirm "
                + expected_confirmation
            )
    v2_stage = (
        "audit" if args.v2_feasibility_audit
        else "run" if args.run_v2
        else None
    )
    repository, upstream = preflight(
        config,
        membership_recovery=recovery_mode,
        v2_stage=v2_stage,
    )
    if args.preflight:
        print("D9 预检：通过")
        print(f"当前窗口模式：{repository['window_mode']}")
        print(f"frozen-oos：{repository['frozen_oos_commit']}")
        if repository["window_mode"] == "design":
            print("下一步：显式解封后再运行 python scripts/d9_pipeline.py")
        return
    if args.recovery_preflight:
        recovery = repository["membership_recovery"]
        if repository["window_mode"] != "oos":
            raise OOSIntegrityError("受控恢复预检要求 OOS 数据保持已解封状态")
        access = ResearchDataAccess(ROOT)
        calendar = load_open_calendar(
            access, config, end_date=config["windows"]["oos_end"]
        )
        membership_audit = validate_benchmark_membership_metadata(
            access, config, calendar
        )
        print("D9 成分元数据受控恢复预检：通过")
        print(f"当前窗口模式：{repository['window_mode']}")
        print(f"原失败清单 SHA256：{recovery['original_manifest_sha256']}")
        print("原失败发生在收益读取、模型训练与绩效生成之前：是")
        print(
            "真实快照不一致日："
            f"{membership_audit['observed_snapshot_mismatch_days']}；"
            "合成调样日不一致："
            f"{membership_audit['synthetic_event_mismatch_days']}"
        )
        print(
            "下一步：python scripts/d9_pipeline.py "
            "--resume-membership-recovery --confirm "
            f"{config['d9']['membership_recovery']['confirmation']}"
        )
        return
    if args.v2_feasibility_audit:
        if repository["window_mode"] != "oos":
            raise OOSIntegrityError("D9 V2 可行性审计要求 OOS 数据保持已解封状态")
        payload, dates = execute_v2_feasibility_audit(config, repository)
        dates_path = ROOT / config["d9"]["v2"]["feasibility_dates"]
        audit_path = ROOT / config["d9"]["v2"]["feasibility_audit"]
        atomic_parquet(dates, dates_path)
        payload["feasibility_dates"] = str(dates_path.relative_to(ROOT))
        payload["feasibility_dates_sha256"] = sha256(dates_path)
        atomic_json(payload, audit_path)
        failure = payload["frozen_first_failure"]
        print("D9 V2 可行性审计：通过")
        print(
            f"冻结口径首次不可行：信号日 {failure['signal_date']}，"
            f"执行日 {failure['execution_date']}"
        )
        print(
            f"最小总换手 {failure['minimum_total_single_sided_turnover']:.2%}；"
            f"其中强制调样 {failure['mandatory_membership_turnover']:.2%}，"
            f"最小主动换手 {failure['minimum_discretionary_single_sided_turnover']:.2%}"
        )
        print(
            "已验证 V2 全部目标日可行；未运行组合回测，未生成收益、IR 或 NW-t。"
        )
        print(
            "下一步：python scripts/d9_pipeline.py --run-v2 --confirm "
            f"{config['d9']['v2']['confirmation']}"
        )
        return
    if repository["window_mode"] != "oos":
        raise OOSIntegrityError(
            "当前仍为 design 模式；请先按运行说明显式解封"
        )
    if args.resume_membership_recovery:
        recovery = repository["membership_recovery"]
        access = ResearchDataAccess(ROOT)
        calendar = load_open_calendar(
            access, config, end_date=config["windows"]["oos_end"]
        )
        membership_metadata_audit = validate_benchmark_membership_metadata(
            access, config, calendar
        )
        recovery["membership_metadata_preflight"] = membership_metadata_audit
        manifest_path = ROOT / config["d9"]["membership_recovery"][
            "recovery_manifest"
        ]
        manifest = {
            "status": "started",
            "recovery_attempt_number": 1,
            "oos_run_number": 1,
            "statistical_oos_run_number": 1,
            "original_manifest": recovery["original_manifest"],
            "original_manifest_sha256": recovery["original_manifest_sha256"],
            "original_failure": {
                "d9_head_commit": recovery["failed_d9_head_commit"],
                "config_sha256": recovery["failed_config_sha256"],
                "error_type": recovery["error_type"],
                "error": recovery["error"],
            },
            "frozen_oos_commit": repository["frozen_oos_commit"],
            "recovery_d9_head_commit": repository["d9_head_commit"],
            "recovery_config_sha256": sha256(ROOT / "config/config.yaml"),
            "scope": "benchmark_membership_metadata_only",
            "performance_observed_before_original_failure": False,
            "model_or_threshold_parameters_changed": False,
            "original_manifest_preserved": True,
            "membership_metadata_preflight": membership_metadata_audit,
            "oos_start": config["windows"]["oos_start"],
            "oos_end": config["windows"]["oos_end"],
            "message": "同一次统计性 OOS 的受控工程恢复已启动；不得再次恢复",
        }
    elif args.run_v2:
        manifest_path = ROOT / config["d9"]["v2"]["run_manifest"]
        source = repository["v2_source_failures"]
        manifest = {
            "status": "started",
            "version": "D9-V2",
            "classification": "post_oos_methodology_amendment",
            "v2_run_number": 1,
            "statistical_oos_run_number": 1,
            "rank_ic_observed_before_v2": True,
            "portfolio_performance_observed_before_v2": False,
            "original_manifest": source["original_manifest"],
            "original_manifest_sha256": source["original_manifest_sha256"],
            "recovery_manifest": source["recovery_manifest"],
            "recovery_manifest_sha256": source["recovery_manifest_sha256"],
            "feasibility_audit": source["feasibility_audit"],
            "feasibility_audit_sha256": source["feasibility_audit_sha256"],
            "source_manifests_preserved": True,
            "frozen_oos_commit": repository["frozen_oos_commit"],
            "v2_head_commit": repository["d9_head_commit"],
            "v2_config_sha256": sha256(ROOT / "config/config.yaml"),
            "changed_rule": (
                "15% 上限约束主动单边换手；强制成分退出仍计入总成交和成本"
            ),
            "model_or_threshold_parameters_changed": False,
            "oos_start": config["windows"]["oos_start"],
            "oos_end": config["windows"]["oos_end"],
            "message": (
                "D9-V2 事后方法修订已启动；必须与原两次失败并列披露，"
                "不得表述为原预注册 D9 的无条件验证"
            ),
        }
    else:
        manifest_path = ROOT / config["d9"]["run_manifest"]
        manifest = {
            "status": "started",
            "oos_run_number": 1,
            "frozen_oos_commit": repository["frozen_oos_commit"],
            "d9_head_commit": repository["d9_head_commit"],
            "config_sha256": sha256(ROOT / "config/config.yaml"),
            "oos_start": config["windows"]["oos_start"],
            "oos_end": config["windows"]["oos_end"],
            "message": "正式 OOS 已启动；无论结果好坏都不得改参数重跑",
        }
    atomic_json(manifest, manifest_path)
    try:
        summary = execute_d9(
            config,
            repository,
            upstream,
            variant="v2" if args.run_v2 else "original",
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rerun_allowed": False,
            }
        )
        atomic_json(manifest, manifest_path)
        raise
    manifest.update(
        {
            "status": "completed",
            "rerun_allowed": False,
            "engineering_acceptance_passed": summary["execution_acceptance"][
                "passed"
            ],
            "decision": summary["threshold_evaluation"]["decision"],
            "decision_interpretation": (
                "post_oos_reference_only" if args.run_v2
                else "preregistered_oos"
            ),
            "summary_file": (
                "reports/d9_v2_summary.json" if args.run_v2
                else "reports/d9_summary.json"
            ),
        }
    )
    atomic_json(manifest, manifest_path)
    engineering_status = (
        "通过" if summary["execution_acceptance"]["passed"] else "失败"
    )
    run_label = "D9-V2" if args.run_v2 else "D9"
    print(f"{run_label} 工程验收：{engineering_status}")
    print(
        f"费后年化超额 {summary['performance']['annualized_excess_return']:.2%}；"
        f"IR {summary['performance']['information_ratio']:.3f}；"
        f"NW-t {summary['performance']['newey_west_t_minimum']:.3f}"
    )
    decision_label = "V2 事后参考判定" if args.run_v2 else "当前判定"
    print(f"{decision_label}：{summary['threshold_evaluation']['decision']}")
    report_name = (
        "D9_V2滚动样本外报告.md" if args.run_v2
        else "D9滚动样本外报告.md"
    )
    print(f"报告：{ROOT / config['data']['reports_dir'] / report_name}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from linq_platform.backtest.engine import print_summary, run_backtest
from linq_platform.config import BacktestConfig
from linq_platform.intelligence import run_native_phase4_1
from linq_platform.research.validation import ValidationLimits, print_validation_report, validate_backtest

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_PROJECT = PROJECT_DIR.parent
RUNS_DIR = PROJECT_DIR / "reports" / "v10_native_runs"
REGISTRY_PATH = RUNS_DIR / "run_registry.csv"
LATEST_PATH = RUNS_DIR / "latest_run.json"
CANDLES = PARENT_PROJECT / "data" / "cache" / "EUR_USD_M5.csv"
SETUPS = PARENT_PROJECT / "reports" / "v7" / "EUR_USD_automatic_setups.csv"
PHASE1 = PARENT_PROJECT / "reports" / "v9" / "EUR_USD_market_database.csv"
LEGACY_SELECTED = PARENT_PROJECT / "reports" / "v9_phase4_1" / "EUR_USD_phase4_1_selected_trades.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_label(text: str) -> str:
    return "_".join("".join(c if c.isalnum() else " " for c in text.lower()).split()) or "native_run"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=True, default=str), encoding="utf-8")


def compare_native_to_legacy(native_path: Path, legacy_path: Path) -> dict[str, Any]:
    native = pd.read_csv(native_path)
    legacy = pd.read_csv(legacy_path)
    for frame in (native, legacy):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    keys = ["timestamp", "direction"]
    cols = ["entry_price", "stop_price", "target_price", "probability_1r", "gross_result_r", "net_result_r"]
    merged = native[keys + cols].merge(legacy[keys + cols], on=keys, how="outer", suffixes=("_native", "_legacy"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    tolerance = 1e-10
    per_column = {}
    for col in cols:
        delta = (pd.to_numeric(matched[f"{col}_native"], errors="coerce") - pd.to_numeric(matched[f"{col}_legacy"], errors="coerce")).abs()
        per_column[col] = {"maximum_absolute_difference": float(delta.max()) if len(delta) else None, "rows_with_difference": int((delta > tolerance).sum())}
    exact = len(native) == len(legacy) == len(matched) and all(v["rows_with_difference"] == 0 for v in per_column.values())
    return {
        "passed": bool(exact),
        "native_rows": int(len(native)),
        "legacy_rows": int(len(legacy)),
        "matched_rows": int(len(matched)),
        "only_native": int((merged["_merge"] == "left_only").sum()),
        "only_legacy": int((merged["_merge"] == "right_only").sum()),
        "columns": per_column,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="native_phase4_1_baseline")
    p.add_argument("--slippage-pips", type=float, default=0.10)
    args = p.parse_args()
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ") + "_" + clean_label(args.label)
    run_dir = RUNS_DIR / run_id
    intel_dir = run_dir / "intelligence"
    exec_dir = run_dir / "execution"
    intel_dir.mkdir(parents=True, exist_ok=False)
    exec_dir.mkdir(parents=True)

    print("=" * 108)
    print("LINQ TRADING PLATFORM V10 — NATIVE INTELLIGENCE RESEARCH RUN")
    print("=" * 108)
    print(f"Run ID:                         {run_id}")

    print("\nSTEP 1 OF 4 — GENERATE SIGNALS NATIVELY")
    print("-" * 108)
    intelligence = run_native_phase4_1(CANDLES, SETUPS, PHASE1, intel_dir, args.slippage_pips)
    selected_path = Path(intelligence["files"]["selected"])
    selected_summary = intelligence["summaries"][0]
    print(f"Selected signals:                {intelligence['selected_trades']}")
    print(f"Native gross R:                  {selected_summary['gross_r']:+.3f}R")
    print(f"Native net R:                    {selected_summary['net_r']:+.3f}R")

    print("\nSTEP 2 OF 4 — VERIFY NATIVE/LEGACY SIGNAL PARITY")
    print("-" * 108)
    source_parity = compare_native_to_legacy(selected_path, LEGACY_SELECTED)
    print(f"Native rows:                     {source_parity['native_rows']}")
    print(f"Legacy rows:                     {source_parity['legacy_rows']}")
    print(f"Matched rows:                    {source_parity['matched_rows']}")
    print(f"Exact source parity:             {'PASSED' if source_parity['passed'] else 'FAILED'}")

    print("\nSTEP 3 OF 4 — EXECUTE WITH BACKTRADER")
    print("-" * 108)
    backtest = run_backtest(CANDLES, selected_path, exec_dir, BacktestConfig())
    print_summary(backtest)
    validation = validate_backtest(backtest, ValidationLimits())
    print_validation_report(validation)

    overall_pass = source_parity["passed"] and validation["passed"]
    result = {
        "run_id": run_id,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "PASSED" if overall_pass else "FAILED",
        "native_intelligence": intelligence,
        "native_legacy_source_parity": source_parity,
        "backtest": backtest,
        "execution_validation": validation,
        "hashes": {"candles": sha256(CANDLES), "setups": sha256(SETUPS), "phase1": sha256(PHASE1), "native_selected": sha256(selected_path), "legacy_selected": sha256(LEGACY_SELECTED)},
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "run_summary.json", result)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(LATEST_PATH, result)

    registry_exists = REGISTRY_PATH.exists()
    with REGISTRY_PATH.open("a", newline="", encoding="utf-8") as f:
        fields = ["run_id", "status", "signals", "native_net_r", "backtrader_net_r", "source_parity", "execution_validation", "run_directory"]
        w = csv.DictWriter(f, fieldnames=fields)
        if not registry_exists: w.writeheader()
        w.writerow({"run_id": run_id, "status": result["status"], "signals": intelligence["selected_trades"], "native_net_r": selected_summary["net_r"], "backtrader_net_r": backtest["net_performance"]["net_r"], "source_parity": source_parity["passed"], "execution_validation": validation["passed"], "run_directory": str(run_dir)})

    print("\nSTEP 4 OF 4 — FINAL NATIVE MIGRATION RESULT")
    print("-" * 108)
    print(f"Native/legacy signal parity:     {'PASSED' if source_parity['passed'] else 'FAILED'}")
    print(f"Backtrader validation:           {validation['status']}")
    print(f"Overall migration status:        {result['status']}")
    print(f"Run directory:                   {run_dir}")
    print("=" * 108)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

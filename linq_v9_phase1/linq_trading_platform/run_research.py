from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from linq_platform.backtest.engine import (
    print_summary,
    run_backtest,
)
from linq_platform.config import BacktestConfig
from linq_platform.research.legacy_bridge import (
    run_phase4_pipeline,
)
from linq_platform.research.validation import (
    ValidationLimits,
    print_validation_report,
    validate_backtest,
)


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_PROJECT = PROJECT_DIR.parent

RUNS_DIR = PROJECT_DIR / "reports" / "v10_runs"
REGISTRY_PATH = RUNS_DIR / "run_registry.csv"
LATEST_SUMMARY_PATH = RUNS_DIR / "latest_run.json"

CANDLES = (
    PARENT_PROJECT
    / "data"
    / "cache"
    / "EUR_USD_M5.csv"
)

PHASE4_SIGNALS = (
    PARENT_PROJECT
    / "reports"
    / "v9_phase4_1"
    / "EUR_USD_phase4_1_selected_trades.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a timestamped LINQ V10 research experiment."
        )
    )

    parser.add_argument(
        "--label",
        default="phase4_1_backtrader",
        help=(
            "Human-readable experiment label. "
            "Spaces are converted to underscores."
        ),
    )

    parser.add_argument(
        "--max-net-parity-difference-r",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max-gross-parity-difference-r",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--allow-skipped-signals",
        action="store_true",
    )

    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help=(
            "Return exit code 0 even when validation fails."
        ),
    )

    return parser.parse_args()


def clean_label(label: str) -> str:
    allowed = []

    for character in label.strip().lower():
        if character.isalnum():
            allowed.append(character)
        elif character in {" ", "-", "_"}:
            allowed.append("_")

    cleaned = "".join(allowed)

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_") or "research_run"


def create_run_id(label: str) -> str:
    timestamp = pd.Timestamp.now(
        tz="UTC"
    ).strftime("%Y%m%dT%H%M%SZ")

    return f"{timestamp}_{clean_label(label)}"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )


def append_registry(
    registry_path: Path,
    record: dict[str, Any],
) -> None:
    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "run_id",
        "created_at_utc",
        "label",
        "validation_status",
        "signals",
        "completed_trades",
        "wins",
        "losses",
        "win_rate",
        "gross_r",
        "spread_cost_r",
        "slippage_cost_r",
        "net_r",
        "expectancy_r",
        "profit_factor",
        "maximum_drawdown_r",
        "phase4_net_r",
        "net_parity_difference_r",
        "run_directory",
    ]

    file_exists = registry_path.exists()

    with registry_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                column: record.get(column)
                for column in columns
            }
        )


def print_final_result(
    record: dict[str, Any],
) -> None:
    print("\nFINAL RESEARCH RUN RESULT")
    print("-" * 108)

    fields = [
        ("Run ID", "run_id"),
        ("Validation status", "validation_status"),
        ("Signals", "signals"),
        ("Completed trades", "completed_trades"),
        ("Wins / losses", "wins_losses"),
        ("Win rate", "win_rate_text"),
        ("Gross result", "gross_r_text"),
        ("Spread cost", "spread_cost_text"),
        ("Slippage cost", "slippage_cost_text"),
        ("Net result", "net_r_text"),
        ("Net expectancy", "expectancy_r_text"),
        ("Profit factor", "profit_factor_text"),
        ("Maximum drawdown", "drawdown_text"),
        ("Net parity difference", "parity_text"),
        ("Run directory", "run_directory"),
    ]

    for label, key in fields:
        print(
            f"{label + ':':<32}"
            f"{record[key]}"
        )

    print("=" * 108)


def main() -> int:
    args = parse_arguments()

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = create_run_id(args.label)
    run_dir = RUNS_DIR / run_id

    intelligence_dir = run_dir / "intelligence"
    execution_dir = run_dir / "execution"

    intelligence_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    execution_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 108)
    print(
        "LINQ TRADING PLATFORM V10 — "
        "VERSIONED RESEARCH RUN"
    )
    print("=" * 108)
    print(
        f"Run ID:                         "
        f"{run_id}"
    )
    print(
        f"Run directory:                  "
        f"{run_dir}"
    )

    manifest_path = (
        intelligence_dir
        / "phase4_1_manifest.json"
    )

    print("\nSTEP 1 OF 4 — GENERATE INTELLIGENCE SIGNALS")
    print("-" * 108)

    intelligence_manifest = run_phase4_pipeline(
        parent_project=PARENT_PROJECT,
        output_manifest=manifest_path,
    )

    signal_validation = intelligence_manifest[
        "signal_validation"
    ]

    signal_snapshot = (
        intelligence_dir
        / "EUR_USD_selected_signals.csv"
    )

    shutil.copy2(
        PHASE4_SIGNALS,
        signal_snapshot,
    )

    print(
        f"Signals generated:              "
        f"{signal_validation['rows']}"
    )
    print(
        f"Signal file SHA-256:            "
        f"{signal_validation['sha256']}"
    )

    print("\nSTEP 2 OF 4 — EXECUTE WITH BACKTRADER")
    print("-" * 108)

    config = BacktestConfig()

    backtest_summary = run_backtest(
        candles_path=CANDLES,
        signals_path=signal_snapshot,
        output_dir=execution_dir,
        config=config,
    )

    print_summary(backtest_summary)

    print("\nSTEP 3 OF 4 — APPLY VALIDATION GATE")
    print("-" * 108)

    limits = ValidationLimits(
        maximum_net_parity_difference_r=(
            args.max_net_parity_difference_r
        ),
        maximum_gross_parity_difference_r=(
            args.max_gross_parity_difference_r
        ),
        allow_skipped_signals=(
            args.allow_skipped_signals
        ),
    )

    validation = validate_backtest(
        summary=backtest_summary,
        limits=limits,
    )

    print_validation_report(validation)

    print("\nSTEP 4 OF 4 — SAVE RUN RECORD")
    print("-" * 108)

    net = backtest_summary["net_performance"]
    gross = backtest_summary["gross_performance"]
    parity = backtest_summary["parity"]

    created_at = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    run_summary = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "label": clean_label(args.label),
        "status": validation["status"],
        "intelligence": intelligence_manifest,
        "backtest": backtest_summary,
        "validation": validation,
        "paths": {
            "run_directory": str(run_dir),
            "signal_snapshot": str(signal_snapshot),
            "intelligence_manifest": str(
                manifest_path
            ),
            "execution_directory": str(
                execution_dir
            ),
        },
    }

    run_summary_path = (
        run_dir
        / "run_summary.json"
    )

    validation_path = (
        run_dir
        / "validation_report.json"
    )

    write_json(
        run_summary_path,
        run_summary,
    )
    write_json(
        validation_path,
        validation,
    )
    write_json(
        LATEST_SUMMARY_PATH,
        run_summary,
    )

    record = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "label": clean_label(args.label),
        "validation_status": validation["status"],
        "signals": int(
            backtest_summary["signals_loaded"]
        ),
        "completed_trades": int(
            net["trades"]
        ),
        "wins": int(net["wins"]),
        "losses": int(net["losses"]),
        "win_rate": float(net["win_rate"]),
        "gross_r": float(gross["net_r"]),
        "spread_cost_r": float(
            backtest_summary["spread_cost_r"]
        ),
        "slippage_cost_r": float(
            backtest_summary["slippage_cost_r"]
        ),
        "net_r": float(net["net_r"]),
        "expectancy_r": float(
            net["expectancy_r"]
        ),
        "profit_factor": float(
            net["profit_factor"]
        ),
        "maximum_drawdown_r": float(
            net["maximum_drawdown_r"]
        ),
        "phase4_net_r": float(
            parity["phase4_net_r"]
        ),
        "net_parity_difference_r": float(
            parity["net_parity_difference_r"]
        ),
        "run_directory": str(run_dir),
    }

    append_registry(
        REGISTRY_PATH,
        record,
    )

    printable = {
        **record,
        "wins_losses": (
            f"{record['wins']} / {record['losses']}"
        ),
        "win_rate_text": (
            f"{record['win_rate']:.1%}"
        ),
        "gross_r_text": (
            f"{record['gross_r']:+.3f}R"
        ),
        "spread_cost_text": (
            f"-{record['spread_cost_r']:.3f}R"
        ),
        "slippage_cost_text": (
            f"-{record['slippage_cost_r']:.3f}R"
        ),
        "net_r_text": (
            f"{record['net_r']:+.3f}R"
        ),
        "expectancy_r_text": (
            f"{record['expectancy_r']:+.3f}R"
        ),
        "profit_factor_text": (
            f"{record['profit_factor']:.3f}"
        ),
        "drawdown_text": (
            f"{record['maximum_drawdown_r']:.3f}R"
        ),
        "parity_text": (
            f"{record['net_parity_difference_r']:+.3f}R"
        ),
    }

    print(
        f"Run summary:                    "
        f"{run_summary_path}"
    )
    print(
        f"Validation report:              "
        f"{validation_path}"
    )
    print(
        f"Run registry:                   "
        f"{REGISTRY_PATH}"
    )
    print(
        f"Latest run summary:             "
        f"{LATEST_SUMMARY_PATH}"
    )

    print_final_result(printable)

    if not validation["passed"] and not args.no_fail_exit:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

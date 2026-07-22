from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from linq_platform.backtest.engine import (
    print_summary,
    run_backtest,
)
from linq_platform.config import BacktestConfig
from linq_platform.research.legacy_bridge import (
    run_phase4_pipeline,
)


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_PROJECT = PROJECT_DIR.parent

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

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "v10_pipeline"
)

BACKTEST_OUTPUT = OUTPUT_DIR / "backtrader"
MANIFEST_PATH = OUTPUT_DIR / "phase4_1_manifest.json"
COMBINED_SUMMARY_PATH = OUTPUT_DIR / "combined_summary.json"
SIGNAL_SNAPSHOT_PATH = (
    OUTPUT_DIR
    / "EUR_USD_phase4_1_selected_trades_snapshot.csv"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 108)
    print(
        "LINQ TRADING PLATFORM V10 — "
        "END-TO-END RESEARCH PIPELINE"
    )
    print("=" * 108)

    print("\nSTEP 1 OF 3 — RUN PHASE 4.1 INTELLIGENCE")
    print("-" * 108)

    phase4_manifest = run_phase4_pipeline(
        parent_project=PARENT_PROJECT,
        output_manifest=MANIFEST_PATH,
    )

    validation = phase4_manifest["signal_validation"]

    print(
        f"Phase 4.1 script:               "
        f"{phase4_manifest['phase4_script']}"
    )
    print(
        f"Execution status:               "
        f"{phase4_manifest['status'].upper()}"
    )
    print(
        f"Execution duration:             "
        f"{phase4_manifest['duration_seconds']:.2f} seconds"
    )
    print(
        f"Selected signals generated:     "
        f"{validation['rows']}"
    )
    print(
        f"Long / short signals:           "
        f"{validation['long_signals']} / "
        f"{validation['short_signals']}"
    )
    print(
        f"Phase 4.1 gross R:              "
        f"{validation['gross_r']:+.3f}R"
    )
    print(
        f"Phase 4.1 net R:                "
        f"{validation['net_r']:+.3f}R"
    )

    shutil.copy2(
        PHASE4_SIGNALS,
        SIGNAL_SNAPSHOT_PATH,
    )

    print("\nSTEP 2 OF 3 — RUN BACKTRADER EXECUTION")
    print("-" * 108)

    config = BacktestConfig()

    backtest_summary = run_backtest(
        candles_path=CANDLES,
        signals_path=SIGNAL_SNAPSHOT_PATH,
        output_dir=BACKTEST_OUTPUT,
        config=config,
    )

    print_summary(backtest_summary)

    print("\nSTEP 3 OF 3 — CREATE COMBINED RUN RECORD")
    print("-" * 108)

    combined = {
        "platform": "LINQ Trading Platform V10",
        "pipeline_type": (
            "Phase 4.1 intelligence plus Backtrader execution"
        ),
        "created_at_utc": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
        "intelligence_engine": phase4_manifest,
        "execution_engine": backtest_summary,
        "reproducibility": {
            "phase4_script_sha256": phase4_manifest[
                "phase4_script_sha256"
            ],
            "candle_file_sha256": phase4_manifest[
                "candle_file_sha256"
            ],
            "selected_signal_sha256": validation[
                "sha256"
            ],
        },
        "files": {
            "signal_snapshot": str(
                SIGNAL_SNAPSHOT_PATH
            ),
            "phase4_manifest": str(
                MANIFEST_PATH
            ),
            "backtrader_summary": str(
                BACKTEST_OUTPUT
                / "EUR_USD_v10_summary.json"
            ),
        },
    }

    COMBINED_SUMMARY_PATH.write_text(
        json.dumps(
            combined,
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )

    parity = backtest_summary["parity"]
    net = backtest_summary["net_performance"]

    print(
        f"Signal snapshot:                "
        f"{SIGNAL_SNAPSHOT_PATH}"
    )
    print(
        f"Combined summary:               "
        f"{COMBINED_SUMMARY_PATH}"
    )

    print("\nFINAL PIPELINE RESULT")
    print("-" * 108)
    print(
        f"Intelligence signals:           "
        f"{validation['rows']}"
    )
    print(
        f"Backtrader completed trades:    "
        f"{net['trades']}"
    )
    print(
        f"Phase 4.1 net result:           "
        f"{parity['phase4_net_r']:+.3f}R"
    )
    print(
        f"Backtrader net result:          "
        f"{parity['backtrader_net_r']:+.3f}R"
    )
    print(
        f"Net parity difference:          "
        f"{parity['net_parity_difference_r']:+.3f}R"
    )
    print(
        f"Backtrader net expectancy:      "
        f"{net['expectancy_r']:+.3f}R"
    )
    print(
        f"Backtrader profit factor:       "
        f"{net['profit_factor']:.3f}"
    )

    print("=" * 108)


if __name__ == "__main__":
    main()

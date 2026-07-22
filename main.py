from pathlib import Path

from linq_platform.backtest.engine import (
    print_summary,
    run_backtest,
)
from linq_platform.config import BacktestConfig


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_PROJECT = PROJECT_DIR.parent

CANDLES = (
    PARENT_PROJECT
    / "data"
    / "cache"
    / "EUR_USD_M5.csv"
)

SIGNALS = (
    PARENT_PROJECT
    / "reports"
    / "v9_phase4_1"
    / "EUR_USD_phase4_1_selected_trades.csv"
)

OUTPUT = PROJECT_DIR / "reports" / "v10"


def main():
    config = BacktestConfig()

    summary = run_backtest(
        candles_path=CANDLES,
        signals_path=SIGNALS,
        output_dir=OUTPUT,
        config=config,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()

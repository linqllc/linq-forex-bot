from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

REGISTRY_PATH = (
    PROJECT_DIR
    / "reports"
    / "v10_runs"
    / "run_registry.csv"
)


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "No research registry exists yet.\n"
            "Run: python run_research.py"
        )

    registry = pd.read_csv(REGISTRY_PATH)

    if registry.empty:
        print("The research registry is empty.")
        return

    columns = [
        "run_id",
        "validation_status",
        "signals",
        "completed_trades",
        "wins",
        "losses",
        "win_rate",
        "net_r",
        "expectancy_r",
        "profit_factor",
        "maximum_drawdown_r",
        "net_parity_difference_r",
    ]

    available = [
        column
        for column in columns
        if column in registry.columns
    ]

    print("=" * 140)
    print("LINQ V10 — RESEARCH RUN REGISTRY")
    print("=" * 140)
    print(
        registry[available]
        .tail(20)
        .to_string(index=False)
    )
    print("=" * 140)


if __name__ == "__main__":
    main()

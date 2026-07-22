#!/usr/bin/env python3
"""Rank experiment performance across walk-forward windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from linq_platform.research.walk_forward import (
    RobustnessConfig,
    evaluate_walk_forward_robustness,
    rank_robust_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and rank experiment results across walk-forward validation windows."
        )
    )

    parser.add_argument(
        "window_results",
        type=Path,
        help=("CSV containing one row per experiment and walk-forward window."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/walk_forward_robustness"),
    )

    parser.add_argument(
        "--minimum-windows",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minimum-trades-per-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minimum-profitable-window-ratio",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-average-expectancy-r",
        type=float,
        default=0.0,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    window_results = pd.read_csv(args.window_results)

    config = RobustnessConfig(
        minimum_windows=args.minimum_windows,
        minimum_trades_per_window=(args.minimum_trades_per_window),
        minimum_profitable_window_ratio=(args.minimum_profitable_window_ratio),
        minimum_average_expectancy_r=(args.minimum_average_expectancy_r),
    )

    robustness = evaluate_walk_forward_robustness(
        window_results,
        config,
    )

    ranked = rank_robust_experiments(
        robustness,
        eligible_only=False,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = args.output / "walk_forward_robustness.csv"

    json_path = args.output / "walk_forward_robustness.json"

    ranked.to_csv(
        csv_path,
        index=False,
    )

    payload = {
        "source_file": str(args.window_results),
        "config": {
            "minimum_windows": (config.minimum_windows),
            "minimum_trades_per_window": (config.minimum_trades_per_window),
            "minimum_profitable_window_ratio": (config.minimum_profitable_window_ratio),
            "minimum_average_expectancy_r": (config.minimum_average_expectancy_r),
        },
        "experiments": ranked.to_dict(orient="records"),
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    eligible = ranked.loc[ranked["eligible"]]

    print("=" * 72)
    print("LINQ WALK-FORWARD ROBUSTNESS")
    print("=" * 72)
    print(f"Experiments analyzed: {len(ranked)}")
    print(f"Eligible experiments: {len(eligible)}")
    print(f"CSV report:           {csv_path}")
    print(f"JSON report:          {json_path}")

    if not eligible.empty:
        print()
        print("TOP ROBUST EXPERIMENTS")
        print(eligible.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from src.research_v10.causal_replay import CausalReplayEngine
from src.research_v10.config import ResearchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ Forex V11.1 causal bar-by-bar replay engine"
        )
    )

    parser.add_argument(
        "--instrument",
        default="EUR_USD",
    )
    parser.add_argument(
        "--candles",
        required=True,
    )
    parser.add_argument(
        "--historical-setups",
        default=None,
    )
    parser.add_argument(
        "--report-dir",
        default="reports/v11_causal",
    )
    parser.add_argument(
        "--retrace-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--session",
        default="london_open",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ResearchConfig(
        instrument=args.instrument,
    )

    engine = CausalReplayEngine(
        config=config,
        retrace_threshold=args.retrace_threshold,
        required_session=args.session,
    )

    summary = engine.run(
        csv_path=args.candles,
        report_dir=args.report_dir,
        historical_setups_path=args.historical_setups,
    )

    print("=" * 100)
    print("LINQ FOREX CAUSAL REPLAY ENGINE — V11.1")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

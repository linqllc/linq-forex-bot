from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research_v10.replay import run_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LINQ Forex V11 sequential replay engine"
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
        "--setups",
        required=True,
    )
    parser.add_argument(
        "--report-dir",
        default="reports/v11",
    )
    parser.add_argument(
        "--rule",
        default=(
            "retrace_fraction le 0.25000000000050465 "
            "AND session eq london_open"
        ),
    )
    parser.add_argument(
        "--minimum-minutes-between-trades",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--one-trade-per-session",
        action="store_true",
    )
    parser.add_argument(
        "--causality-checks",
        type=int,
        default=12,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 100)
    print("LINQ FOREX REPLAY ENGINE — V11")
    print("=" * 100)

    summary = run_replay(
        setups_csv=Path(args.setups),
        candles_csv=Path(args.candles),
        report_dir=Path(args.report_dir),
        instrument=args.instrument,
        rule=args.rule,
        minimum_minutes_between_trades=(
            args.minimum_minutes_between_trades
        ),
        one_trade_per_session=args.one_trade_per_session,
        causality_checks=args.causality_checks,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

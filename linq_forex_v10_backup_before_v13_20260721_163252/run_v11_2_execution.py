from __future__ import annotations

import argparse
import json

from src.research_v10.config import ResearchConfig
from src.research_v10.execution_replay import (
    ExecutionConfig,
    RealisticExecutionReplay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ Forex V11.2 execution-realism replay"
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
        "--report-dir",
        default="reports/v11_2_execution",
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

    parser.add_argument(
        "--spread-pips",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--entry-slippage-pips",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--exit-slippage-pips",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--entry-delay-bars",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-open-trades",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--minimum-minutes-between-entries",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--one-trade-per-session",
        action="store_true",
    )
    parser.add_argument(
        "--allow-repeat-impulse",
        action="store_true",
    )
    parser.add_argument(
        "--reject-if-entry-gap-atr",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-stop-pips",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=2_000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    research_config = ResearchConfig(
        instrument=args.instrument,
    )

    execution_config = ExecutionConfig(
        spread_pips=args.spread_pips,
        entry_slippage_pips=(
            args.entry_slippage_pips
        ),
        exit_slippage_pips=(
            args.exit_slippage_pips
        ),
        entry_delay_bars=args.entry_delay_bars,
        max_open_trades=args.max_open_trades,
        one_trade_per_impulse=(
            not args.allow_repeat_impulse
        ),
        one_trade_per_session=(
            args.one_trade_per_session
        ),
        minimum_minutes_between_entries=(
            args.minimum_minutes_between_entries
        ),
        reject_if_entry_gap_atr=(
            args.reject_if_entry_gap_atr
        ),
        minimum_stop_pips=(
            args.minimum_stop_pips
        ),
        account_starting_balance=(
            args.starting_balance
        ),
        monte_carlo_runs=(
            args.monte_carlo_runs
        ),
    )

    engine = RealisticExecutionReplay(
        research_config=research_config,
        execution_config=execution_config,
        retrace_threshold=(
            args.retrace_threshold
        ),
        required_session=args.session,
    )

    summary = engine.run(
        csv_path=args.candles,
        report_dir=args.report_dir,
    )

    print("=" * 100)
    print(
        "LINQ FOREX EXECUTION REALISM — V11.2"
    )
    print("=" * 100)
    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

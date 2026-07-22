from __future__ import annotations

import argparse
import json

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.pending_limit_replay import (
    PendingLimitConfig,
    PendingLimitReplay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ Forex V11.3 causal "
            "pending-limit replay"
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
        default="reports/v11_3_pending_limit",
    )

    parser.add_argument(
        "--retrace-fraction",
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
        "--limit-slippage-pips",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--exit-slippage-pips",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-open-trades",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-pending-orders",
        type=int,
        default=20,
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
        "--cancel-outside-session",
        action="store_true",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    research_config = ResearchConfig(
        instrument=args.instrument,
    )

    limit_config = PendingLimitConfig(
        retrace_fraction=(
            args.retrace_fraction
        ),
        required_session=args.session,
        spread_pips=args.spread_pips,
        limit_slippage_pips=(
            args.limit_slippage_pips
        ),
        exit_slippage_pips=(
            args.exit_slippage_pips
        ),
        max_open_trades=(
            args.max_open_trades
        ),
        max_pending_orders=(
            args.max_pending_orders
        ),
        one_trade_per_impulse=(
            not args.allow_repeat_impulse
        ),
        one_trade_per_session=(
            args.one_trade_per_session
        ),
        cancel_outside_required_session=(
            args.cancel_outside_session
        ),
        minimum_stop_pips=(
            args.minimum_stop_pips
        ),
        account_starting_balance=(
            args.starting_balance
        ),
    )

    engine = PendingLimitReplay(
        research_config=research_config,
        limit_config=limit_config,
    )

    summary = engine.run(
        csv_path=args.candles,
        report_dir=args.report_dir,
    )

    print("=" * 100)
    print(
        "LINQ FOREX PENDING-LIMIT "
        "EXECUTION — V11.3"
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

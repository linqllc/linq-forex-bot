from __future__ import annotations

import argparse
import json

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.execution_compare import (
    ExactSignalExecutionComparison,
    ExecutionScenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ V12 frozen-signal "
            "execution comparison"
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
        default=(
            "reports/v12_execution_compare"
        ),
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
        "--expected-accepted",
        type=int,
        default=498,
    )

    parser.add_argument(
        "--spread-pips",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--minimum-stop-pips",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    research_config = ResearchConfig(
        instrument=args.instrument,
    )

    scenarios = [
        ExecutionScenario(
            name="ideal_signal_price",
            entry_mode="signal_price",
        ),
        ExecutionScenario(
            name="signal_close_no_cost",
            entry_mode="signal_close",
        ),
        ExecutionScenario(
            name="signal_close_spread_only",
            entry_mode="signal_close",
            spread_pips=args.spread_pips,
        ),
        ExecutionScenario(
            name="signal_close_full_cost",
            entry_mode="signal_close",
            spread_pips=args.spread_pips,
            entry_slippage_pips=(
                args.slippage_pips
            ),
            exit_slippage_pips=(
                args.slippage_pips
            ),
        ),
        ExecutionScenario(
            name="signal_close_latency_0_1",
            entry_mode="signal_close",
            spread_pips=args.spread_pips,
            entry_slippage_pips=(
                args.slippage_pips
            ),
            exit_slippage_pips=(
                args.slippage_pips
            ),
            latency_pips=0.1,
        ),
        ExecutionScenario(
            name="signal_close_latency_0_2",
            entry_mode="signal_close",
            spread_pips=args.spread_pips,
            entry_slippage_pips=(
                args.slippage_pips
            ),
            exit_slippage_pips=(
                args.slippage_pips
            ),
            latency_pips=0.2,
        ),
        ExecutionScenario(
            name="signal_close_latency_0_5",
            entry_mode="signal_close",
            spread_pips=args.spread_pips,
            entry_slippage_pips=(
                args.slippage_pips
            ),
            exit_slippage_pips=(
                args.slippage_pips
            ),
            latency_pips=0.5,
        ),
        ExecutionScenario(
            name="next_bar_open_full_cost",
            entry_mode="next_open",
            spread_pips=args.spread_pips,
            entry_slippage_pips=(
                args.slippage_pips
            ),
            exit_slippage_pips=(
                args.slippage_pips
            ),
        ),
    ]

    engine = ExactSignalExecutionComparison(
        research_config=research_config,
        retrace_threshold=(
            args.retrace_threshold
        ),
        required_session=args.session,
        expected_accepted_signals=(
            args.expected_accepted
        ),
        minimum_stop_pips=(
            args.minimum_stop_pips
        ),
    )

    summary = engine.run(
        csv_path=args.candles,
        report_dir=args.report_dir,
        scenarios=scenarios,
    )

    print("=" * 112)
    print(
        "LINQ V12 — FROZEN-SIGNAL "
        "EXECUTION COMPARISON"
    )
    print("=" * 112)

    frozen = summary[
        "frozen_signal_population"
    ]

    print(
        f"All causal signals:       "
        f"{frozen['all_signals']}"
    )

    print(
        f"Accepted frozen signals: "
        f"{frozen['accepted_signals']}"
    )

    print(
        f"Expected accepted:        "
        f"{frozen['expected_accepted_signals']}"
    )

    print(
        f"Signal count match:       "
        f"{frozen['count_match']}"
    )

    print(
        f"Unique setup IDs:         "
        f"{frozen['unique_setup_ids']}"
    )

    print(
        f"Signal identity hash:     "
        f"{frozen['identity_hash']}"
    )

    print("\n" + "-" * 112)

    comparison = summary["comparison"]

    for row in comparison:
        print(
            f"{row['scenario']:<32}"
            f" trades={row['trades']:>4}"
            f"  WR={row['win_rate'] * 100:>6.2f}%"
            f"  Exp={row['expectancy_r']:>8.4f}R"
            f"  PF={row['profit_factor']:>7.3f}"
            f"  DD={row['max_drawdown_r']:>9.2f}R"
            f"  Total={row['total_r']:>9.2f}R"
        )

    print("-" * 112)

    print(
        "\nFull summary:\n"
        + json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

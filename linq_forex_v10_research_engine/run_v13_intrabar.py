from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.intrabar_reconstruction import (
    CausalIntrabarReconstruction,
    IntrabarScenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ V13 causal M5-to-M1 "
            "intrabar reconstruction"
        )
    )

    parser.add_argument(
        "--instrument",
        default="EUR_USD",
    )
    parser.add_argument(
        "--m5",
        required=True,
    )
    parser.add_argument(
        "--m1",
        required=True,
    )
    parser.add_argument(
        "--report-dir",
        default="reports/v13_intrabar",
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
        "--slippage-pips",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--timestamps-are-close",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    m1_path = Path(args.m1)

    if not m1_path.exists():
        raise FileNotFoundError(
            f"M1 candle file not found: {m1_path}"
        )

    config = ResearchConfig(
        instrument=args.instrument,
    )

    common = {
        "spread_pips": args.spread_pips,
        "entry_slippage_pips": (
            args.slippage_pips
        ),
        "exit_slippage_pips": (
            args.slippage_pips
        ),
    }

    scenarios = [
        IntrabarScenario(
            name="limit_retrace_10",
            entry_mode="limit_touch",
            retrace_fraction=0.10,
            **common,
        ),
        IntrabarScenario(
            name="limit_retrace_15",
            entry_mode="limit_touch",
            retrace_fraction=0.15,
            **common,
        ),
        IntrabarScenario(
            name="limit_retrace_20",
            entry_mode="limit_touch",
            retrace_fraction=0.20,
            **common,
        ),
        IntrabarScenario(
            name="limit_retrace_25",
            entry_mode="limit_touch",
            retrace_fraction=0.25,
            **common,
        ),
        IntrabarScenario(
            name="m1_reversal_max_10",
            entry_mode="reversal_confirm",
            max_retrace_fraction=0.10,
            **common,
        ),
        IntrabarScenario(
            name="m1_reversal_max_15",
            entry_mode="reversal_confirm",
            max_retrace_fraction=0.15,
            **common,
        ),
        IntrabarScenario(
            name="m1_reversal_max_20",
            entry_mode="reversal_confirm",
            max_retrace_fraction=0.20,
            **common,
        ),
        IntrabarScenario(
            name="m1_reversal_max_25",
            entry_mode="reversal_confirm",
            max_retrace_fraction=0.25,
            **common,
        ),
    ]

    engine = CausalIntrabarReconstruction(
        config=config,
        required_session=args.session,
        timestamps_are_bar_open=(
            not args.timestamps_are_close
        ),
    )

    summary = engine.run(
        m5_path=args.m5,
        m1_path=args.m1,
        report_dir=args.report_dir,
        scenarios=scenarios,
    )

    print("=" * 122)
    print(
        "LINQ V13 — CAUSAL M5→M1 "
        "INTRABAR RECONSTRUCTION"
    )
    print("=" * 122)

    print(
        f"M5 candles:  "
        f"{summary['m5_candles']}"
    )
    print(
        f"M1 candles:  "
        f"{summary['m1_candles']}"
    )
    print(
        f"M5 impulses: "
        f"{summary['m5_impulses']}"
    )

    print("\n" + "-" * 122)

    for row in summary["comparison"]:
        win_rate = row["win_rate"]
        expectancy = row["expectancy_r"]
        profit_factor = row["profit_factor"]

        wr_text = (
            f"{win_rate * 100:6.2f}%"
            if win_rate is not None
            else "   N/A "
        )
        exp_text = (
            f"{expectancy:8.4f}R"
            if expectancy is not None
            else "     N/A"
        )
        pf_text = (
            f"{profit_factor:7.3f}"
            if profit_factor is not None
            else "    N/A"
        )

        print(
            f"{row['scenario']:<28}"
            f" trades={row['trades']:>5}"
            f" fill={row['fill_rate'] * 100:>6.2f}%"
            f" WR={wr_text}"
            f" Exp={exp_text}"
            f" PF={pf_text}"
            f" DD={row['max_drawdown_r']!s:>10}"
            f" Total={row['total_r']!s:>10}"
        )

    print("-" * 122)

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

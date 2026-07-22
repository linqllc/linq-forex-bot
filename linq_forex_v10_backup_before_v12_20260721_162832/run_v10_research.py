from __future__ import annotations

import argparse
import json

from src.research_v10 import ForexResearchEngine, ResearchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LINQ V10 Forex single-strategy research engine"
    )
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report-dir", default="reports/v10")
    parser.add_argument("--timestamp-column", default=None)
    parser.add_argument("--pip-size", type=float, default=0.0001)
    parser.add_argument("--displacement-atr-min", type=float, default=1.25)
    parser.add_argument("--displacement-body-ratio-min", type=float, default=0.65)
    parser.add_argument("--target-r", type=float, default=1.5)
    parser.add_argument("--stop-atr", type=float, default=1.25)
    parser.add_argument("--max-holding-bars", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResearchConfig(
        instrument=args.instrument,
        timestamp_column=args.timestamp_column,
        pip_size=args.pip_size,
        displacement_atr_min=args.displacement_atr_min,
        displacement_body_ratio_min=args.displacement_body_ratio_min,
        target_r=max(1.0, args.target_r),
        stop_atr=args.stop_atr,
        max_holding_bars=args.max_holding_bars,
    )
    summary = ForexResearchEngine(config).run(args.csv, args.report_dir)

    print("=" * 100)
    print("LINQ FOREX RESEARCH ENGINE — V10")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

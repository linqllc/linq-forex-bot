from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.automatic_supply_demand import (
    detect_automatic_zones,
    generate_automatic_setups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LINQ v7 automatic supply-and-demand detector"
        )
    )

    parser.add_argument(
        "--instrument",
        default="EUR_USD",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Historical candle CSV. Defaults to the cached "
            "OANDA candle file for the selected instrument."
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/v7",
    )
    parser.add_argument(
        "--body-multiplier",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--impulse-atr",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--minimum-impulse-candles",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--maximum-zone-touches",
        type=int,
        default=1,
    )

    return parser.parse_args()


def resolve_csv(args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv)

    candidates = [
        Path(
            f"data/processed/"
            f"{args.instrument}_oanda_candles.csv"
        ),
        Path(
            f"data/cache/"
            f"{args.instrument}_oanda_candles.csv"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "No cached candle CSV found. Run v5 with real OANDA data "
        "first, or provide --csv /path/to/candles.csv."
    )


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv(args)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Reading candles: {csv_path}")
    candles = pd.read_csv(csv_path)

    print("Detecting automatic supply and demand zones...")

    features, zones = detect_automatic_zones(
        candles,
        instrument=args.instrument,
        body_multiplier=args.body_multiplier,
        impulse_atr_multiplier=args.impulse_atr,
        minimum_impulse_candles=(
            args.minimum_impulse_candles
        ),
        maximum_zone_touches=args.maximum_zone_touches,
    )

    print("Generating confirmation setups...")

    setups = generate_automatic_setups(
        features,
        zones,
    )

    zones_path = output / (
        f"{args.instrument}_automatic_zones.csv"
    )
    setups_path = output / (
        f"{args.instrument}_automatic_setups.csv"
    )
    features_path = output / (
        f"{args.instrument}_detection_features.csv"
    )

    zones.to_csv(zones_path, index=False)
    setups.to_csv(setups_path, index=False)
    features.to_csv(features_path, index=False)

    grade_counts = (
        setups["grade"].value_counts().to_dict()
        if not setups.empty
        else {}
    )

    summary = {
        "instrument": args.instrument,
        "candle_rows": len(candles),
        "zones_detected": len(zones),
        "valid_zones": (
            int((~zones["invalidated"]).sum())
            if not zones.empty
            else 0
        ),
        "fresh_zones": (
            int(zones["fresh"].sum())
            if not zones.empty
            else 0
        ),
        "setups_generated": len(setups),
        "grade_counts": grade_counts,
        "average_confluence_score": (
            float(setups["confluence_score"].mean())
            if not setups.empty
            else None
        ),
    }

    summary_path = output / (
        f"{args.instrument}_automatic_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print()
    print(f"Zones:   {zones_path}")
    print(f"Setups:  {setups_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

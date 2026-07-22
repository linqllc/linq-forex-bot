from pathlib import Path

import pandas as pd

from src.automatic_supply_demand import detect_automatic_zones
from src.multi_timeframe import add_multi_timeframe_context


CANDLES_PATH = Path("data/cache/EUR_USD_M5.csv")
OUTPUT_PATH = Path("reports/v7/EUR_USD_rejection_diagnostics.txt")
INSTRUMENT = "EUR_USD"
CONFIRMATION_LOOKAHEAD = 100


def main() -> None:
    candles = pd.read_csv(CANDLES_PATH)
    candles = add_multi_timeframe_context(candles)

    features, zones = detect_automatic_zones(
        candles,
        instrument=INSTRUMENT,
        body_multiplier=1.2,
        impulse_atr_multiplier=1.5,
        minimum_impulse_candles=3,
        maximum_zone_touches=1,
    )

    counters = {
        "zones_total": len(zones),
        "zones_invalidated_before_scan": 0,
        "zones_scanned": 0,
        "never_returned_to_zone": 0,
        "returned_to_zone": 0,
        "zone_invalidated_on_return": 0,
        "h1_aligned_returns": 0,
        "h1_countertrend_returns": 0,
        "aligned_without_confirmation": 0,
        "aligned_with_confirmation": 0,
    }

    rows = []

    for _, zone in zones.iterrows():
        if bool(zone["invalidated"]):
            counters["zones_invalidated_before_scan"] += 1
            continue

        counters["zones_scanned"] += 1

        direction = (
            "long"
            if zone["zone_type"] == "demand"
            else "short"
        )

        start = int(zone["impulse_end_index"]) + 1
        end = min(
            len(features),
            start + CONFIRMATION_LOOKAHEAD,
        )

        returned = False

        for index in range(start, end):
            row = features.iloc[index]

            entered_zone = (
                float(row["mid_low"]) <= float(zone["zone_top"])
                and float(row["mid_high"]) >= float(zone["zone_bottom"])
            )

            if not entered_zone:
                continue

            returned = True
            counters["returned_to_zone"] += 1

            if direction == "long":
                zone_invalid = (
                    float(row["mid_close"])
                    < float(zone["zone_bottom"])
                )
                h1_aligned = row.get("h1_trend") == "bullish"
                confirmation = bool(row["bullish"])
            else:
                zone_invalid = (
                    float(row["mid_close"])
                    > float(zone["zone_top"])
                )
                h1_aligned = row.get("h1_trend") == "bearish"
                confirmation = bool(row["bearish"])

            if zone_invalid:
                counters["zone_invalidated_on_return"] += 1
                rows.append(
                    {
                        "zone_id": zone["zone_id"],
                        "direction": direction,
                        "return_time": row["time"],
                        "h1_trend": row.get("h1_trend"),
                        "result": "zone_invalidated_on_return",
                    }
                )
                break

            if h1_aligned:
                counters["h1_aligned_returns"] += 1

                if confirmation:
                    counters["aligned_with_confirmation"] += 1
                    result = "aligned_with_confirmation"
                else:
                    counters["aligned_without_confirmation"] += 1
                    result = "aligned_without_confirmation"
            else:
                counters["h1_countertrend_returns"] += 1
                result = "countertrend_return"

            rows.append(
                {
                    "zone_id": zone["zone_id"],
                    "direction": direction,
                    "return_time": row["time"],
                    "h1_trend": row.get("h1_trend"),
                    "h4_trend": row.get("h4_trend"),
                    "confirmation": confirmation,
                    "result": result,
                }
            )

            if h1_aligned and confirmation:
                break

        if not returned:
            counters["never_returned_to_zone"] += 1

    lines = [
        "V7 EUR/USD Rejection Diagnostics",
        "=" * 55,
        "",
    ]

    for key, value in counters.items():
        lines.append(f"{key}: {value}")

    report = "\n".join(lines)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)

    detail_path = OUTPUT_PATH.with_name(
        "EUR_USD_rejection_details.csv"
    )
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    print(report)
    print(f"\nSaved summary to: {OUTPUT_PATH}")
    print(f"Saved details to: {detail_path}")


if __name__ == "__main__":
    main()

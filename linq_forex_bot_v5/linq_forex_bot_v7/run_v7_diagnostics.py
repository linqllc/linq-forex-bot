from pathlib import Path

import pandas as pd


SETUPS_PATH = Path("reports/v7/EUR_USD_automatic_setups.csv")
ZONES_PATH = Path("reports/v7/EUR_USD_automatic_zones.csv")
OUTPUT_PATH = Path("reports/v7/EUR_USD_diagnostics.txt")


def percent(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{value / total:.1%}"


def main() -> None:
    if not SETUPS_PATH.exists():
        raise FileNotFoundError(f"Missing setups file: {SETUPS_PATH}")

    if not ZONES_PATH.exists():
        raise FileNotFoundError(f"Missing zones file: {ZONES_PATH}")

    setups = pd.read_csv(SETUPS_PATH)
    zones = pd.read_csv(ZONES_PATH)

    lines: list[str] = []

    lines.append("V7 EUR/USD Diagnostics")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Total zones detected: {len(zones)}")
    lines.append(f"Total setups generated: {len(setups)}")
    lines.append("")

    if not zones.empty:
        lines.append("ZONE BREAKDOWN")
        lines.append("-" * 50)

        for column in [
            "zone_type",
            "valid",
            "fresh",
            "touches",
        ]:
            if column in zones.columns:
                lines.append(f"\n{column}:")
                counts = zones[column].value_counts(dropna=False)
                for value, count in counts.items():
                    lines.append(
                        f"  {value}: {count} ({percent(int(count), len(zones))})"
                    )

    if not setups.empty:
        lines.append("")
        lines.append("SETUP FILTER PASS RATES")
        lines.append("-" * 50)

        filters = [
            "fresh_zone",
            "has_fvg",
            "break_of_structure",
            "higher_timeframe_aligned",
            "slow_pullback",
            "confirmation_candle",
            "discount_or_premium",
            "zone_rejection",
        ]

        for column in filters:
            if column not in setups.columns:
                continue

            values = setups[column].fillna(False).astype(bool)
            passed = int(values.sum())

            lines.append(
                f"{column}: {passed}/{len(setups)} "
                f"({percent(passed, len(setups))})"
            )

        if "grade" in setups.columns:
            lines.append("")
            lines.append("GRADE DISTRIBUTION")
            lines.append("-" * 50)

            for grade, count in setups["grade"].value_counts().items():
                lines.append(
                    f"{grade}: {count} ({percent(int(count), len(setups))})"
                )

        if "confluence_score" in setups.columns:
            lines.append("")
            lines.append("CONFLUENCE SCORES")
            lines.append("-" * 50)
            lines.append(
                f"Minimum: {setups['confluence_score'].min():.1f}"
            )
            lines.append(
                f"Maximum: {setups['confluence_score'].max():.1f}"
            )
            lines.append(
                f"Average: {setups['confluence_score'].mean():.1f}"
            )
            lines.append(
                f"Median: {setups['confluence_score'].median():.1f}"
            )

    report = "\n".join(lines)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)

    print(report)
    print(f"\nSaved diagnostics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

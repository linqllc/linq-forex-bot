from pathlib import Path

import pandas as pd

from src.setup_outcomes import analyze_outcomes


CANDLES_PATH = Path("data/cache/EUR_USD_M5.csv")
SETUPS_PATH = Path("reports/v7/EUR_USD_automatic_setups.csv")
OUTPUT_PATH = Path("reports/v7/EUR_USD_outcomes.csv")


def normalize_candle_columns(candles: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "time": (
            "time",
            "timestamp",
            "datetime",
            "date",
        ),
        "open": (
            "open",
            "o",
            "mid_o",
            "bid_o",
            "ask_o",
            "mid_open",
        ),
        "high": (
            "high",
            "h",
            "mid_h",
            "bid_h",
            "ask_h",
            "mid_high",
        ),
        "low": (
            "low",
            "l",
            "mid_l",
            "bid_l",
            "ask_l",
            "mid_low",
        ),
        "close": (
            "close",
            "c",
            "mid_c",
            "bid_c",
            "ask_c",
            "mid_close",
        ),
    }

    lower_lookup = {
        str(column).strip().lower(): column
        for column in candles.columns
    }

    rename_map = {}

    for standard_name, possible_names in candidates.items():
        matched_column = None

        for possible_name in possible_names:
            if possible_name in lower_lookup:
                matched_column = lower_lookup[possible_name]
                break

        if matched_column is None:
            raise ValueError(
                f"Could not identify '{standard_name}' column.\n"
                f"Available columns: {candles.columns.tolist()}"
            )

        rename_map[matched_column] = standard_name

    return candles.rename(columns=rename_map)


def main() -> None:
    if not CANDLES_PATH.exists():
        raise FileNotFoundError(f"Missing candle file: {CANDLES_PATH}")

    if not SETUPS_PATH.exists():
        raise FileNotFoundError(f"Missing setup file: {SETUPS_PATH}")

    candles = pd.read_csv(CANDLES_PATH)
    setups = pd.read_csv(SETUPS_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if setups.empty:
        previous_output_removed = OUTPUT_PATH.exists()
        OUTPUT_PATH.unlink(missing_ok=True)

        print("\nAnalyzed 0 setups")
        print("No H1-aligned setups were generated.")
        if previous_output_removed:
            print(f"Removed stale outcomes file: {OUTPUT_PATH}")
        return

    candles = normalize_candle_columns(candles)

    results = analyze_outcomes(candles, setups)
    results.to_csv(OUTPUT_PATH, index=False)

    display_columns = [
        "setup_id",
        "timestamp",
        "direction",
        "grade",
        "confluence_score",
        "outcome",
        "mfe_r",
        "mae_r",
        "hit_1r",
        "hit_2r",
        "hit_3r",
        "stop_hit",
        "bars_evaluated",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    print(f"\nAnalyzed {len(results)} setups\n")

    if results.empty:
        print("No setups were available to evaluate.")
    else:
        print(results[available_columns].to_string(index=False))

    print(f"\nSaved outcomes to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

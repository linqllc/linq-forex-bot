from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import pandas as pd


REPORT_DIR = Path("reports/v7")

SETUPS_PATH = REPORT_DIR / "EUR_USD_automatic_setups.csv"
OUTCOMES_PATH = REPORT_DIR / "EUR_USD_outcomes.csv"

JSON_OUTPUT = REPORT_DIR / "EUR_USD_factor_analysis.json"
FACTOR_OUTPUT = REPORT_DIR / "EUR_USD_factor_rankings.csv"
COMBO_OUTPUT = REPORT_DIR / "EUR_USD_filter_combinations.csv"


BINARY_FEATURES = [
    "fresh_zone",
    "has_fvg",
    "break_of_structure",
    "higher_timeframe_aligned",
    "slow_pullback",
    "confirmation_candle",
    "discount_or_premium",
    "zone_rejection",
]


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "hit_1r_pct": 0.0,
            "hit_2r_pct": 0.0,
            "hit_3r_pct": 0.0,
            "stop_rate_pct": 0.0,
            "average_mfe_r": 0.0,
            "average_mae_r": 0.0,
        }

    realized = df["realized_r"]

    gross_profit = float(realized[realized > 0].sum())
    gross_loss = abs(float(realized[realized < 0].sum()))

    if gross_loss == 0:
        profit_factor = "inf" if gross_profit > 0 else 0.0
    else:
        profit_factor = round(gross_profit / gross_loss, 3)

    return {
        "trades": int(len(df)),
        "wins": int((realized > 0).sum()),
        "losses": int((realized < 0).sum()),
        "win_rate_pct": round(float((realized > 0).mean() * 100), 1),
        "total_r": round(float(realized.sum()), 3),
        "expectancy_r": round(float(realized.mean()), 3),
        "profit_factor": profit_factor,
        "hit_1r_pct": round(float(df["hit_1r"].mean() * 100), 1),
        "hit_2r_pct": round(float(df["hit_2r"].mean() * 100), 1),
        "hit_3r_pct": round(float(df["hit_3r"].mean() * 100), 1),
        "stop_rate_pct": round(float(df["stop_hit"].mean() * 100), 1),
        "average_mfe_r": round(float(df["mfe_r"].mean()), 3),
        "average_mae_r": round(float(df["mae_r"].mean()), 3),
    }


def print_stats(label: str, stats: dict) -> None:
    print(
        f"{label:<38}"
        f" trades={stats['trades']:>3}"
        f" wins={stats['wins']:>3}"
        f" exp={stats['expectancy_r']:>7}"
        f" total={stats['total_r']:>8}R"
        f" PF={str(stats['profit_factor']):>6}"
        f" 3R={stats['hit_3r_pct']:>5}%"
        f" stop={stats['stop_rate_pct']:>5}%"
    )


def load_data() -> pd.DataFrame:
    missing_files = [
        str(path)
        for path in [SETUPS_PATH, OUTCOMES_PATH]
        if not path.exists()
    ]

    if missing_files:
        raise SystemExit(
            "Missing required files:\n"
            + "\n".join(f"- {path}" for path in missing_files)
        )

    setups = pd.read_csv(SETUPS_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    required_setup_columns = [
        "setup_id",
        "timestamp",
        "direction",
        "grade",
        "confluence_score",
    ]

    required_outcome_columns = [
        "setup_id",
        "mfe_r",
        "mae_r",
        "hit_1r",
        "hit_2r",
        "hit_3r",
        "stop_hit",
    ]

    missing_setup_columns = [
        column
        for column in required_setup_columns
        if column not in setups.columns
    ]

    missing_outcome_columns = [
        column
        for column in required_outcome_columns
        if column not in outcomes.columns
    ]

    if missing_setup_columns:
        raise SystemExit(
            "Setup CSV is missing columns: "
            + ", ".join(missing_setup_columns)
        )

    if missing_outcome_columns:
        raise SystemExit(
            "Outcome CSV is missing columns: "
            + ", ".join(missing_outcome_columns)
        )

    outcome_columns = required_outcome_columns.copy()

    if "bars_evaluated" in outcomes.columns:
        outcome_columns.append("bars_evaluated")

    df = setups.merge(
        outcomes[outcome_columns],
        on="setup_id",
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise SystemExit("No matching setup and outcome rows were found.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"]).copy()

    for column in [
        "hit_1r",
        "hit_2r",
        "hit_3r",
        "stop_hit",
    ]:
        df[column] = normalize_bool(df[column])

    for feature in BINARY_FEATURES:
        if feature in df.columns:
            df[feature] = normalize_bool(df[feature])

    for column in [
        "confluence_score",
        "mfe_r",
        "mae_r",
        "touches",
        "bars_evaluated",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Correct fixed-3R strategy result:
    # +3R only when a trade reaches 3R.
    # Every trade that fails to reach 3R is treated as -1R.
    df["realized_r"] = -1.0
    df.loc[df["hit_3r"], "realized_r"] = 3.0

    df["hour_utc"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    return df


def analyze_grades(df: pd.DataFrame) -> dict:
    results = {}

    print("\nGRADE PERFORMANCE")
    print("=" * 115)

    for grade in ["A", "B+", "B", "C+", "C"]:
        subset = df[df["grade"] == grade]

        if subset.empty:
            continue

        stats = summarize(subset)
        results[grade] = stats
        print_stats(grade, stats)

    return results


def analyze_thresholds(df: pd.DataFrame) -> dict:
    results = {}

    print("\nCONFLUENCE THRESHOLDS")
    print("=" * 115)

    thresholds = sorted(
        set(
            [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
            + [
                int(value)
                for value in df["confluence_score"].dropna().unique()
            ]
        )
    )

    for threshold in thresholds:
        subset = df[df["confluence_score"] >= threshold]

        if subset.empty:
            continue

        stats = summarize(subset)
        results[str(threshold)] = stats
        print_stats(f"Score >= {threshold}", stats)

    return results


def analyze_direction(df: pd.DataFrame) -> dict:
    results = {}

    print("\nDIRECTION PERFORMANCE")
    print("=" * 115)

    for direction in sorted(df["direction"].dropna().unique()):
        subset = df[df["direction"] == direction]
        stats = summarize(subset)
        results[str(direction)] = stats
        print_stats(str(direction), stats)

    return results


def analyze_features(
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    results = {}
    ranking_rows = []

    print("\nBINARY FACTOR PERFORMANCE")
    print("=" * 115)

    for feature in BINARY_FEATURES:
        if feature not in df.columns:
            continue

        present = df[df[feature]]
        absent = df[~df[feature]]

        present_stats = summarize(present)
        absent_stats = summarize(absent)

        if absent.empty:
            delta = None
        else:
            delta = round(
                present_stats["expectancy_r"]
                - absent_stats["expectancy_r"],
                3,
            )

        results[feature] = {
            "present": present_stats,
            "absent": absent_stats,
            "expectancy_delta_r": delta,
        }

        print(f"\n{feature}")
        print_stats("Present", present_stats)
        print_stats("Absent", absent_stats)

        if delta is None:
            print("Expectancy comparison unavailable: no absent trades")
        else:
            print(f"Expectancy improvement: {delta:+.3f}R")

        ranking_rows.append(
            {
                "factor": feature,
                "trades_present": present_stats["trades"],
                "trades_absent": absent_stats["trades"],
                "expectancy_present_r": present_stats["expectancy_r"],
                "expectancy_absent_r": absent_stats["expectancy_r"],
                "expectancy_delta_r": delta,
                "profit_factor_present": present_stats["profit_factor"],
                "hit_3r_present_pct": present_stats["hit_3r_pct"],
                "stop_rate_present_pct": present_stats["stop_rate_pct"],
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    if not ranking.empty:
        ranking = ranking.sort_values(
            "expectancy_delta_r",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return results, ranking


def analyze_touches(df: pd.DataFrame) -> dict:
    results = {}

    if "touches" not in df.columns:
        return results

    print("\nZONE TOUCH PERFORMANCE")
    print("=" * 115)

    groups = {
        "0": df[df["touches"] == 0],
        "1": df[df["touches"] == 1],
        "2": df[df["touches"] == 2],
        "3+": df[df["touches"] >= 3],
    }

    for label, subset in groups.items():
        if subset.empty:
            continue

        stats = summarize(subset)
        results[label] = stats
        print_stats(f"Touches {label}", stats)

    return results


def analyze_time(df: pd.DataFrame) -> dict:
    results = {
        "hour_utc": {},
        "weekday": {},
    }

    print("\nENTRY HOUR PERFORMANCE — UTC")
    print("=" * 115)

    for hour in range(24):
        subset = df[df["hour_utc"] == hour]

        if len(subset) < 2:
            continue

        stats = summarize(subset)
        results["hour_utc"][str(hour)] = stats
        print_stats(f"{hour:02d}:00 UTC", stats)

    print("\nDAY-OF-WEEK PERFORMANCE")
    print("=" * 115)

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for weekday in weekday_order:
        subset = df[df["weekday"] == weekday]

        if subset.empty:
            continue

        stats = summarize(subset)
        results["weekday"][weekday] = stats
        print_stats(weekday, stats)

    return results


def analyze_stop_behavior(df: pd.DataFrame) -> dict:
    stopped = df[df["stop_hit"]].copy()

    print("\nENTRY AND STOP BEHAVIOR")
    print("=" * 115)

    thresholds = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    reached = {}

    for threshold in thresholds:
        count = int((stopped["mfe_r"] >= threshold).sum())
        percentage = (
            round(count / len(stopped) * 100, 1)
            if len(stopped)
            else 0.0
        )

        key = str(threshold).replace(".", "_") + "r"

        reached[key] = {
            "count": count,
            "percentage": percentage,
        }

        print(
            f"Stopped after reaching {threshold:>4.2f}R:"
            f" {count:>3}/{len(stopped):<3}"
            f" ({percentage:>5.1f}%)"
        )

    return {
        "stopped_trades": int(len(stopped)),
        "average_stopped_trade_mfe_r": (
            round(float(stopped["mfe_r"].mean()), 3)
            if len(stopped)
            else 0.0
        ),
        "average_stopped_trade_mae_r": (
            round(float(stopped["mae_r"].mean()), 3)
            if len(stopped)
            else 0.0
        ),
        "profit_excursion_before_stop": reached,
    }


def analyze_exit_models(df: pd.DataFrame) -> dict:
    results = {}

    print("\nSIMPLIFIED FIXED-TARGET EXIT MODELS")
    print("=" * 115)

    for target in [1.0, 1.5, 2.0, 3.0]:
        realized = pd.Series(-1.0, index=df.index)
        realized.loc[df["mfe_r"] >= target] = target

        wins = int((realized > 0).sum())
        gross_profit = float(realized[realized > 0].sum())
        gross_loss = abs(float(realized[realized < 0].sum()))

        profit_factor = (
            round(gross_profit / gross_loss, 3)
            if gross_loss > 0
            else "inf"
        )

        stats = {
            "target_r": target,
            "trades": int(len(df)),
            "wins": wins,
            "win_rate_pct": round(wins / len(df) * 100, 1),
            "total_r": round(float(realized.sum()), 3),
            "expectancy_r": round(float(realized.mean()), 3),
            "profit_factor": profit_factor,
        }

        results[str(target)] = stats

        print(
            f"Fixed {target:<3}R"
            f" trades={stats['trades']:>3}"
            f" wins={stats['wins']:>3}"
            f" win_rate={stats['win_rate_pct']:>5}%"
            f" expectancy={stats['expectancy_r']:>7}R"
            f" total={stats['total_r']:>8}R"
            f" PF={str(stats['profit_factor']):>6}"
        )

    return results


def analyze_combinations(
    df: pd.DataFrame,
    minimum_trades: int = 5,
) -> pd.DataFrame:
    available = [
        feature
        for feature in BINARY_FEATURES
        if feature in df.columns
        and df[feature].nunique() > 1
    ]

    rows = []

    for size in [2, 3]:
        for combo in combinations(available, size):
            mask = pd.Series(True, index=df.index)

            for feature in combo:
                mask &= df[feature]

            subset = df[mask]

            if len(subset) < minimum_trades:
                continue

            stats = summarize(subset)

            rows.append(
                {
                    "filters": " + ".join(combo),
                    "filter_count": size,
                    "trades": stats["trades"],
                    "wins": stats["wins"],
                    "win_rate_pct": stats["win_rate_pct"],
                    "total_r": stats["total_r"],
                    "expectancy_r": stats["expectancy_r"],
                    "profit_factor": stats["profit_factor"],
                    "hit_3r_pct": stats["hit_3r_pct"],
                    "stop_rate_pct": stats["stop_rate_pct"],
                    "average_mfe_r": stats["average_mfe_r"],
                    "average_mae_r": stats["average_mae_r"],
                }
            )

    result = pd.DataFrame(rows)

    print("\nTOP FILTER COMBINATIONS")
    print("=" * 115)

    if result.empty:
        print("No combinations met the minimum trade requirement.")
        return result

    result = result.sort_values(
        ["expectancy_r", "trades"],
        ascending=[False, False],
    ).reset_index(drop=True)

    print(result.head(20).to_string(index=False))

    return result


def main() -> None:
    df = load_data()

    print("\n")
    print("=" * 115)
    print("V7 EUR/USD CORRECTED FACTOR ANALYSIS")
    print("=" * 115)
    print(f"Trades loaded: {len(df)}")

    overall = summarize(df)
    print_stats("Overall fixed-3R strategy", overall)

    grades = analyze_grades(df)
    thresholds = analyze_thresholds(df)
    directions = analyze_direction(df)
    features, factor_rankings = analyze_features(df)
    touches = analyze_touches(df)
    time_results = analyze_time(df)
    stop_behavior = analyze_stop_behavior(df)
    exit_models = analyze_exit_models(df)
    combinations_df = analyze_combinations(df)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not factor_rankings.empty:
        factor_rankings.to_csv(
            FACTOR_OUTPUT,
            index=False,
        )

        print("\nFACTOR RANKING BY CORRECTED EXPECTANCY")
        print("=" * 115)
        print(factor_rankings.to_string(index=False))

    if not combinations_df.empty:
        combinations_df.to_csv(
            COMBO_OUTPUT,
            index=False,
        )

    payload = {
        "strategy_model": {
            "winner": "+3R when hit_3r is true",
            "loss": "-1R otherwise",
        },
        "overall": overall,
        "grades": grades,
        "thresholds": thresholds,
        "directions": directions,
        "features": features,
        "touches": touches,
        "time": time_results,
        "stop_behavior": stop_behavior,
        "exit_models": exit_models,
        "top_filter_combinations": (
            combinations_df.head(50).to_dict(orient="records")
            if not combinations_df.empty
            else []
        ),
    }

    JSON_OUTPUT.write_text(
        json.dumps(payload, indent=2)
    )

    print("\nFILES SAVED")
    print("=" * 115)
    print(JSON_OUTPUT)

    if FACTOR_OUTPUT.exists():
        print(FACTOR_OUTPUT)

    if COMBO_OUTPUT.exists():
        print(COMBO_OUTPUT)


if __name__ == "__main__":
    main()

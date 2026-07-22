from pathlib import Path
import json

import pandas as pd

REPORT_DIR = Path("reports/v7")


def summarize(df):
    if len(df) == 0:
        return None

    r = df["outcome"].map({
        "stopped": -1,
        "hit_1r": 1,
        "hit_2r": 2,
        "hit_3r": 3,
    })

    return {
        "trades": len(df),
        "expectancy": round(r.mean(), 2),
        "win_1r": round(df["hit_1r"].mean() * 100, 1),
        "win_2r": round(df["hit_2r"].mean() * 100, 1),
        "win_3r": round(df["hit_3r"].mean() * 100, 1),
    }


setups = pd.read_csv(
    REPORT_DIR / "EUR_USD_automatic_setups.csv"
)

outcomes = pd.read_csv(
    REPORT_DIR / "EUR_USD_outcomes.csv"
)

df = setups.merge(
    outcomes[
        [
            "setup_id",
            "outcome",
            "hit_1r",
            "hit_2r",
            "hit_3r",
            "mfe_r",
            "mae_r",
        ]
    ],
    on="setup_id",
)

print("\n")
print("=" * 80)
print("V7 STRATEGY ANALYZER")
print("=" * 80)

###########################################################################
print("\nGRADE PERFORMANCE")
print("-" * 80)

grade_summary = {}

for grade in ["A", "B+", "B", "C+", "C"]:

    subset = df[df.grade == grade]

    if subset.empty:
        continue

    stats = summarize(subset)
    grade_summary[grade] = stats

    print(
        f"{grade:>2}"
        f" | Trades {stats['trades']:>3}"
        f" | Exp {stats['expectancy']:>5}"
        f" | 1R {stats['win_1r']:>5}%"
        f" | 2R {stats['win_2r']:>5}%"
        f" | 3R {stats['win_3r']:>5}%"
    )

###########################################################################
print("\nCONFLUENCE THRESHOLDS")
print("-" * 80)

threshold_summary = {}

for threshold in range(40, 86, 5):

    subset = df[df.confluence_score >= threshold]

    if subset.empty:
        continue

    stats = summarize(subset)
    threshold_summary[str(threshold)] = stats

    print(
        f">={threshold}"
        f" | Trades {stats['trades']:>3}"
        f" | Exp {stats['expectancy']:>5}"
        f" | 3R {stats['win_3r']:>5}%"
    )

###########################################################################
print("\nLONG VS SHORT")
print("-" * 80)

direction_summary = {}

for direction in ["long", "short"]:

    subset = df[df.direction == direction]

    stats = summarize(subset)

    direction_summary[direction] = stats

    print(
        f"{direction:<5}"
        f" | Trades {stats['trades']:>3}"
        f" | Exp {stats['expectancy']:>5}"
        f" | 3R {stats['win_3r']:>5}%"
    )

###########################################################################
print("\nFEATURE PERFORMANCE")
print("-" * 80)

feature_summary = {}

features = [
    "fresh_zone",
    "has_fvg",
    "break_of_structure",
    "discount_or_premium",
    "zone_rejection",
]

for feature in features:

    print(f"\n{feature}")

    feature_summary[feature] = {}

    for value in [True, False]:

        subset = df[df[feature] == value]

        if subset.empty:
            continue

        stats = summarize(subset)

        feature_summary[feature][str(value)] = stats

        print(
            f"  {value:<5}"
            f" Trades {stats['trades']:>3}"
            f" Exp {stats['expectancy']:>5}"
            f" 3R {stats['win_3r']:>5}%"
        )

###########################################################################
print("\nTOP 20 HIGHEST CONFLUENCE TRADES")
print("-" * 80)

cols = [
    "timestamp",
    "direction",
    "grade",
    "confluence_score",
    "outcome",
    "mfe_r",
    "mae_r",
]

print(
    df.sort_values(
        "confluence_score",
        ascending=False,
    )[cols].head(20).to_string(index=False)
)

###########################################################################

summary = {
    "grade": grade_summary,
    "thresholds": threshold_summary,
    "direction": direction_summary,
    "features": feature_summary,
}

output = REPORT_DIR / "EUR_USD_strategy_analysis.json"

output.write_text(
    json.dumps(summary, indent=2)
)

print("\nSaved:", output)

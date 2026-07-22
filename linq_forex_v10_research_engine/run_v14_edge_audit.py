from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "impulse_timestamp",
    "impulse_time",
    "signal_timestamp",
    "signal_time",
    "setup_timestamp",
    "setup_time",
    "decision_timestamp",
    "decision_time",
    "entry_timestamp",
    "entry_time",
    "timestamp",
    "time",
    "datetime",
]

ENTRY_CANDIDATES = [
    "historical_entry_price",
    "original_entry_price",
    "ideal_entry_price",
    "planned_entry_price",
    "entry_price",
    "entry",
]

OUTCOME_CANDIDATES = [
    "realized_r",
    "net_r",
    "result_r",
    "outcome_r",
    "pnl_r",
    "r_multiple",
    "r",
]

DIRECTION_CANDIDATES = [
    "direction",
    "side",
    "trade_direction",
    "signal_direction",
]

ID_CANDIDATES = [
    "impulse_id",
    "setup_id",
    "signal_id",
    "trade_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LINQ V14 signal and execution edge audit."
    )

    parser.add_argument("--v11", required=True)
    parser.add_argument("--v13-trades", required=True)
    parser.add_argument("--v13-impulses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--match-tolerance-minutes",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def first_existing(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def detect_time_column(
    df: pd.DataFrame,
    label: str,
) -> str:
    column = first_existing(df, TIME_CANDIDATES)

    if column is not None:
        return column

    for candidate in df.columns:
        name = str(candidate).lower()

        if any(
            token in name
            for token in [
                "time",
                "date",
                "timestamp",
            ]
        ):
            parsed = pd.to_datetime(
                df[candidate],
                utc=True,
                errors="coerce",
            )

            if parsed.notna().mean() >= 0.80:
                return candidate

    raise RuntimeError(
        f"Could not identify a timestamp column in {label}.\n"
        f"Columns: {list(df.columns)}"
    )


def normalize_direction(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {
        "long",
        "buy",
        "bull",
        "bullish",
        "1",
        "1.0",
    }:
        return "long"

    if text in {
        "short",
        "sell",
        "bear",
        "bearish",
        "-1",
        "-1.0",
    }:
        return "short"

    return text


def load_csv(path: str) -> pd.DataFrame:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(source)

    return pd.read_csv(source)


def prepare_frame(
    df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    result = df.copy()

    time_column = detect_time_column(
        result,
        label,
    )

    result["_audit_time"] = pd.to_datetime(
        result[time_column],
        utc=True,
        errors="coerce",
    )

    result = (
        result[result["_audit_time"].notna()]
        .sort_values("_audit_time")
        .reset_index(drop=True)
    )

    direction_column = first_existing(
        result,
        DIRECTION_CANDIDATES,
    )

    if direction_column:
        result["_audit_direction"] = result[
            direction_column
        ].map(normalize_direction)
    else:
        result["_audit_direction"] = None

    metadata = {
        "label": label,
        "rows": int(len(result)),
        "time_column": time_column,
        "direction_column": direction_column,
        "entry_column": first_existing(
            result,
            ENTRY_CANDIDATES,
        ),
        "outcome_column": first_existing(
            result,
            OUTCOME_CANDIDATES,
        ),
        "id_column": first_existing(
            result,
            ID_CANDIDATES,
        ),
        "columns": [
            str(column)
            for column in result.columns
            if not str(column).startswith("_audit")
        ],
    }

    return result, metadata


def merge_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance_minutes: float,
    suffix: str,
) -> pd.DataFrame:
    right_columns = [
        column
        for column in right.columns
        if column != "_audit_direction"
    ]

    renamed = right[right_columns].rename(
        columns={
            column: f"{column}{suffix}"
            for column in right_columns
            if column != "_audit_time"
        }
    )

    merged = pd.merge_asof(
        left.sort_values("_audit_time"),
        renamed.sort_values("_audit_time"),
        on="_audit_time",
        direction="nearest",
        tolerance=pd.Timedelta(
            minutes=tolerance_minutes
        ),
    )

    return merged


def suspicious_columns(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    patterns = [
        r"future",
        r"forward",
        r"next_",
        r"outcome",
        r"result",
        r"target_hit",
        r"stop_hit",
        r"mfe",
        r"mae",
        r"label",
        r"winner",
        r"profitable",
        r"post_",
        r"subsequent",
    ]

    rows = []

    for frame_name, frame in frames.items():
        for column in frame.columns:
            name = str(column).lower()

            matches = [
                pattern
                for pattern in patterns
                if re.search(pattern, name)
            ]

            if matches:
                rows.append(
                    {
                        "dataset": frame_name,
                        "column": column,
                        "matched_patterns": ",".join(
                            matches
                        ),
                    }
                )

    return pd.DataFrame(rows)


def timestamp_order_audit(
    frame: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    time_columns = []

    for column in frame.columns:
        name = str(column).lower()

        if any(
            token in name
            for token in [
                "time",
                "timestamp",
                "datetime",
            ]
        ):
            parsed = pd.to_datetime(
                frame[column],
                utc=True,
                errors="coerce",
            )

            if parsed.notna().mean() >= 0.50:
                time_columns.append(column)

    rows = []

    for left_column in time_columns:
        left = pd.to_datetime(
            frame[left_column],
            utc=True,
            errors="coerce",
        )

        for right_column in time_columns:
            if left_column == right_column:
                continue

            right = pd.to_datetime(
                frame[right_column],
                utc=True,
                errors="coerce",
            )

            valid = left.notna() & right.notna()

            if not valid.any():
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "left_column": left_column,
                    "right_column": right_column,
                    "comparable_rows": int(
                        valid.sum()
                    ),
                    "left_after_right_rows": int(
                        (left[valid] > right[valid]).sum()
                    ),
                    "left_before_right_rows": int(
                        (left[valid] < right[valid]).sum()
                    ),
                    "equal_rows": int(
                        (left[valid] == right[valid]).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_price_advantage(
    matched: pd.DataFrame,
    v11_entry_column: str | None,
    v13_entry_column: str | None,
    direction_column: str | None,
) -> pd.DataFrame:
    result = matched.copy()

    if (
        v11_entry_column is None
        or v13_entry_column is None
    ):
        result["price_advantage_pips"] = np.nan
        return result

    v13_name = f"{v13_entry_column}_v13"

    if v13_name not in result.columns:
        result["price_advantage_pips"] = np.nan
        return result

    historical = pd.to_numeric(
        result[v11_entry_column],
        errors="coerce",
    )

    causal = pd.to_numeric(
        result[v13_name],
        errors="coerce",
    )

    direction = (
        result["_audit_direction"]
        if "_audit_direction" in result.columns
        else pd.Series(
            None,
            index=result.index,
        )
    )

    advantage = np.where(
        direction.eq("long"),
        causal - historical,
        np.where(
            direction.eq("short"),
            historical - causal,
            np.nan,
        ),
    )

    result["price_advantage_pips"] = (
        advantage / 0.0001
    )

    return result


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_v11 = load_csv(args.v11)
    raw_v13_trades = load_csv(
        args.v13_trades
    )
    raw_v13_impulses = load_csv(
        args.v13_impulses
    )

    v11, meta_v11 = prepare_frame(
        raw_v11,
        "v11_accepted",
    )

    v13_trades, meta_v13_trades = prepare_frame(
        raw_v13_trades,
        "v13_trades",
    )

    v13_impulses, meta_v13_impulses = prepare_frame(
        raw_v13_impulses,
        "v13_impulses",
    )

    match_base = merge_nearest(
        v11,
        v13_impulses,
        args.match_tolerance_minutes,
        "_impulse",
    )

    matched_to_trades = merge_nearest(
        v11,
        v13_trades,
        args.match_tolerance_minutes,
        "_v13",
    )

    matched_to_trades = calculate_price_advantage(
        matched_to_trades,
        meta_v11["entry_column"],
        meta_v13_trades["entry_column"],
        meta_v11["direction_column"],
    )

    v13_time_name = "_audit_time"

    matched_impulses = int(
        match_base[
            [
                column
                for column in match_base.columns
                if column.endswith("_impulse")
            ][0]
        ].notna().sum()
    ) if any(
        column.endswith("_impulse")
        for column in match_base.columns
    ) else 0

    trade_match_columns = [
        column
        for column in matched_to_trades.columns
        if column.endswith("_v13")
    ]

    matched_trades = (
        int(
            matched_to_trades[
                trade_match_columns[0]
            ].notna().sum()
        )
        if trade_match_columns
        else 0
    )

    v11_outcome = meta_v11["outcome_column"]
    v13_outcome = meta_v13_trades[
        "outcome_column"
    ]

    outcome_summary = {}

    if (
        v11_outcome
        and v13_outcome
        and f"{v13_outcome}_v13"
        in matched_to_trades.columns
    ):
        historical_r = pd.to_numeric(
            matched_to_trades[v11_outcome],
            errors="coerce",
        )

        causal_r = pd.to_numeric(
            matched_to_trades[
                f"{v13_outcome}_v13"
            ],
            errors="coerce",
        )

        valid = (
            historical_r.notna()
            & causal_r.notna()
        )

        comparison = matched_to_trades.loc[
            valid
        ].copy()

        comparison["v11_r"] = (
            historical_r[valid]
        )

        comparison["v13_r"] = causal_r[valid]

        comparison["outcome_change"] = np.select(
            [
                (comparison["v11_r"] > 0)
                & (comparison["v13_r"] <= 0),
                (comparison["v11_r"] <= 0)
                & (comparison["v13_r"] > 0),
                (comparison["v11_r"] > 0)
                & (comparison["v13_r"] > 0),
            ],
            [
                "v11_win_to_v13_loss",
                "v11_loss_to_v13_win",
                "winner_in_both",
            ],
            default="loser_in_both",
        )

        comparison.to_csv(
            output_dir
            / "v11_v13_outcome_comparison.csv",
            index=False,
        )

        outcome_summary = {
            "comparable_trades": int(
                len(comparison)
            ),
            "v11_mean_r": float(
                comparison["v11_r"].mean()
            ),
            "v13_mean_r": float(
                comparison["v13_r"].mean()
            ),
            "v11_winners_changed_to_v13_losses": int(
                (
                    comparison["outcome_change"]
                    == "v11_win_to_v13_loss"
                ).sum()
            ),
            "v11_losses_changed_to_v13_winners": int(
                (
                    comparison["outcome_change"]
                    == "v11_loss_to_v13_win"
                ).sum()
            ),
            "winner_in_both": int(
                (
                    comparison["outcome_change"]
                    == "winner_in_both"
                ).sum()
            ),
            "loser_in_both": int(
                (
                    comparison["outcome_change"]
                    == "loser_in_both"
                ).sum()
            ),
        }

    suspicious = suspicious_columns(
        {
            "v11_accepted": raw_v11,
            "v13_trades": raw_v13_trades,
            "v13_impulses": raw_v13_impulses,
        }
    )

    timestamp_audit = pd.concat(
        [
            timestamp_order_audit(
                raw_v11,
                "v11_accepted",
            ),
            timestamp_order_audit(
                raw_v13_trades,
                "v13_trades",
            ),
            timestamp_order_audit(
                raw_v13_impulses,
                "v13_impulses",
            ),
        ],
        ignore_index=True,
    )

    matched_to_trades.to_csv(
        output_dir
        / "v11_v13_nearest_trade_matches.csv",
        index=False,
    )

    match_base.to_csv(
        output_dir
        / "v11_v13_impulse_matches.csv",
        index=False,
    )

    suspicious.to_csv(
        output_dir
        / "suspicious_feature_columns.csv",
        index=False,
    )

    timestamp_audit.to_csv(
        output_dir
        / "timestamp_order_audit.csv",
        index=False,
    )

    price_advantage = pd.to_numeric(
        matched_to_trades[
            "price_advantage_pips"
        ],
        errors="coerce",
    ).dropna()

    summary = {
        "engine_version": "14.0-edge-audit",
        "match_tolerance_minutes": (
            args.match_tolerance_minutes
        ),
        "schemas": {
            "v11": meta_v11,
            "v13_trades": meta_v13_trades,
            "v13_impulses": meta_v13_impulses,
        },
        "matching": {
            "v11_rows": int(len(v11)),
            "v11_matched_to_v13_impulses": (
                matched_impulses
            ),
            "v11_matched_to_v13_trades": (
                matched_trades
            ),
        },
        "entry_price_advantage": {
            "observations": int(
                len(price_advantage)
            ),
            "mean_pips": (
                float(price_advantage.mean())
                if len(price_advantage)
                else None
            ),
            "median_pips": (
                float(price_advantage.median())
                if len(price_advantage)
                else None
            ),
            "p90_pips": (
                float(
                    price_advantage.quantile(
                        0.90
                    )
                )
                if len(price_advantage)
                else None
            ),
        },
        "outcome_comparison": outcome_summary,
        "suspicious_feature_column_count": int(
            len(suspicious)
        ),
        "reports": {
            "trade_matches": str(
                output_dir
                / "v11_v13_nearest_trade_matches.csv"
            ),
            "impulse_matches": str(
                output_dir
                / "v11_v13_impulse_matches.csv"
            ),
            "outcome_comparison": str(
                output_dir
                / "v11_v13_outcome_comparison.csv"
            ),
            "suspicious_columns": str(
                output_dir
                / "suspicious_feature_columns.csv"
            ),
            "timestamp_audit": str(
                output_dir
                / "timestamp_order_audit.csv"
            ),
        },
    }

    summary_path = (
        output_dir
        / "v14_edge_audit_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    print("=" * 100)
    print("LINQ V14 — EDGE AUDIT")
    print("=" * 100)

    print("\nDetected schemas:")
    print(json.dumps(
        summary["schemas"],
        indent=2,
    ))

    print("\nMatching:")
    print(json.dumps(
        summary["matching"],
        indent=2,
    ))

    print("\nEntry-price advantage:")
    print(json.dumps(
        summary["entry_price_advantage"],
        indent=2,
    ))

    print("\nOutcome comparison:")
    print(json.dumps(
        summary["outcome_comparison"],
        indent=2,
    ))

    print(
        "\nSuspicious feature columns:",
        len(suspicious),
    )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()

"""
Native candidate-generation engine.

These functions are extracted directly from the validated LINQ market-memory
reference implementation. Their behavior is protected by native-versus-legacy
parity tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .market_memory import Config


__all__ = [
    "calculate_pivots",
    "first_threshold_hit",
    "detect_reactions",
    "cluster_direction_reactions",
    "build_zones",
    "candle_touches_zone",
    "find_first_zone_touch",
]


def calculate_pivots(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)

    pivot_low = np.zeros(len(df), dtype=bool)
    pivot_high = np.zeros(len(df), dtype=bool)

    for index in range(left, len(df) - right):
        low_window = lows[index - left : index + right + 1]
        high_window = highs[index - left : index + right + 1]

        pivot_low[index] = lows[index] <= np.min(low_window)
        pivot_high[index] = highs[index] >= np.max(high_window)

    return pivot_low, pivot_high


def first_threshold_hit(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    upper_price: float,
    lower_price: float,
) -> tuple[str | None, int | None]:
    """
    Determines which threshold is reached first.

    Conservative same-candle rule:
    If both levels are touched in the same candle, return "both".
    """

    for index in range(start_index, end_index + 1):
        candle_high = float(df.at[index, "high"])
        candle_low = float(df.at[index, "low"])

        upper_hit = candle_high >= upper_price
        lower_hit = candle_low <= lower_price

        if upper_hit and lower_hit:
            return "both", index

        if upper_hit:
            return "upper", index

        if lower_hit:
            return "lower", index

    return None, None


def detect_reactions(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    pivot_low, pivot_high = calculate_pivots(
        df,
        config.pivot_left_bars,
        config.pivot_right_bars,
    )

    reactions: list[dict] = []

    last_possible_index = len(df) - config.reaction_lookahead_bars - 1

    for index in range(
        config.pivot_left_bars,
        last_possible_index,
    ):
        atr = float(df.at[index, "atr"])

        if not math.isfinite(atr) or atr <= 0:
            continue

        future_start = index + 1
        future_end = min(
            index + config.reaction_lookahead_bars,
            len(df) - 1,
        )

        # -------------------------------------------------------------
        # Demand reaction:
        # Price forms a pivot low, then moves upward by the required
        # ATR distance before failing downward by the same amount.
        # -------------------------------------------------------------
        if pivot_low[index]:
            anchor = float(df.at[index, "low"])

            upper_threshold = anchor + config.reaction_atr * atr

            lower_failure = anchor - config.reaction_atr * atr

            result, hit_index = first_threshold_hit(
                df,
                future_start,
                future_end,
                upper_threshold,
                lower_failure,
            )

            if result == "upper" and hit_index is not None:
                future_slice = df.loc[index + 1 : future_end]

                maximum_departure = float(future_slice["high"].max()) - anchor

                reactions.append(
                    {
                        "reaction_index": index,
                        "reaction_time": df.at[index, "time"],
                        "confirmation_index": hit_index,
                        "confirmation_time": df.at[hit_index, "time"],
                        "direction": "demand",
                        "anchor_price": anchor,
                        "distal_price": float(df.at[index, "low"]),
                        "proximal_price": float(df.at[index, "body_high"]),
                        "atr": atr,
                        "bars_to_confirmation": hit_index - index,
                        "maximum_departure": maximum_departure,
                        "departure_atr": maximum_departure / atr,
                    }
                )

        # -------------------------------------------------------------
        # Supply reaction:
        # Price forms a pivot high, then moves downward by the required
        # ATR distance before failing upward by the same amount.
        # -------------------------------------------------------------
        if pivot_high[index]:
            anchor = float(df.at[index, "high"])

            lower_threshold = anchor - config.reaction_atr * atr

            upper_failure = anchor + config.reaction_atr * atr

            result, hit_index = first_threshold_hit(
                df,
                future_start,
                future_end,
                upper_failure,
                lower_threshold,
            )

            if result == "lower" and hit_index is not None:
                future_slice = df.loc[index + 1 : future_end]

                maximum_departure = anchor - float(future_slice["low"].min())

                reactions.append(
                    {
                        "reaction_index": index,
                        "reaction_time": df.at[index, "time"],
                        "confirmation_index": hit_index,
                        "confirmation_time": df.at[hit_index, "time"],
                        "direction": "supply",
                        "anchor_price": anchor,
                        "distal_price": float(df.at[index, "high"]),
                        "proximal_price": float(df.at[index, "body_low"]),
                        "atr": atr,
                        "bars_to_confirmation": hit_index - index,
                        "maximum_departure": maximum_departure,
                        "departure_atr": maximum_departure / atr,
                    }
                )

    reactions_df = pd.DataFrame(reactions)

    if reactions_df.empty:
        return reactions_df

    reactions_df = reactions_df.sort_values(["reaction_time", "direction"]).reset_index(drop=True)

    return reactions_df


def cluster_direction_reactions(
    reactions: pd.DataFrame,
    direction: str,
    config: Config,
) -> list[pd.DataFrame]:
    subset = reactions[reactions["direction"] == direction].copy()

    if subset.empty:
        return []

    subset = subset.sort_values("anchor_price").reset_index(drop=True)

    clusters: list[list[int]] = []
    current_cluster: list[int] = [0]

    for row_index in range(1, len(subset)):
        current = subset.iloc[row_index]
        previous = subset.iloc[current_cluster[-1]]

        local_atr = float(
            np.nanmedian(
                [
                    current["atr"],
                    previous["atr"],
                ]
            )
        )

        allowed_distance = config.cluster_radius_atr * local_atr

        actual_distance = abs(float(current["anchor_price"]) - float(previous["anchor_price"]))

        if actual_distance <= allowed_distance:
            current_cluster.append(row_index)
        else:
            clusters.append(current_cluster)
            current_cluster = [row_index]

    clusters.append(current_cluster)

    return [subset.iloc[indexes].copy() for indexes in clusters]


def build_zones(
    reactions: pd.DataFrame,
    as_of_time: pd.Timestamp,
    config: Config,
) -> pd.DataFrame:
    """
    Uses only reactions confirmed by as_of_time.

    This prevents a reaction from being used before its future move had
    actually occurred.
    """

    lookback_start = as_of_time - pd.Timedelta(days=config.zone_lookback_days)

    eligible = reactions[
        (reactions["reaction_time"] >= lookback_start)
        & (reactions["confirmation_time"] <= as_of_time)
    ].copy()

    if eligible.empty:
        return pd.DataFrame()

    zones: list[dict] = []

    for direction in ["demand", "supply"]:
        clusters = cluster_direction_reactions(
            eligible,
            direction,
            config,
        )

        for cluster_number, cluster in enumerate(clusters, start=1):
            if len(cluster) < config.minimum_reactions:
                continue

            median_atr = float(cluster["atr"].median())

            if direction == "demand":
                # Distal = far/lower boundary.
                # Proximal = near/upper boundary.
                distal = float(cluster["distal_price"].quantile(0.10))
                proximal = float(cluster["proximal_price"].quantile(0.75))

                if proximal <= distal:
                    proximal = float(cluster["anchor_price"].max())

            else:
                # Supply:
                # Proximal = near/lower boundary.
                # Distal = far/upper boundary.
                proximal = float(cluster["proximal_price"].quantile(0.25))
                distal = float(cluster["distal_price"].quantile(0.90))

                if distal <= proximal:
                    distal = float(cluster["anchor_price"].max())

            zone_low = min(proximal, distal)
            zone_high = max(proximal, distal)
            zone_width = zone_high - zone_low

            if median_atr <= 0:
                continue

            width_atr = zone_width / median_atr

            if width_atr > config.maximum_zone_width_atr:
                continue

            reaction_count = int(len(cluster))
            average_departure_atr = float(cluster["departure_atr"].mean())
            median_speed = float(cluster["bars_to_confirmation"].median())

            speed_score = 1.0 / max(median_speed, 1.0)

            latest_reaction_time = cluster["reaction_time"].max()
            age_days = max(
                (as_of_time - latest_reaction_time).total_seconds() / 86_400,
                0.0,
            )

            recency_score = math.exp(-age_days / config.zone_lookback_days)

            strength_score = (
                reaction_count * average_departure_atr * (1.0 + speed_score) * recency_score
            )

            zones.append(
                {
                    "zone_id": (
                        f"{direction}_{cluster_number}_{as_of_time.strftime('%Y%m%d%H%M')}"
                    ),
                    "direction": direction,
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "proximal": proximal,
                    "distal": distal,
                    "zone_width": zone_width,
                    "width_atr": width_atr,
                    "median_atr": median_atr,
                    "reaction_count": reaction_count,
                    "average_departure_atr": average_departure_atr,
                    "median_confirmation_bars": median_speed,
                    "first_reaction_time": cluster["reaction_time"].min(),
                    "latest_reaction_time": latest_reaction_time,
                    "strength_score": strength_score,
                    "as_of_time": as_of_time,
                }
            )

    zones_df = pd.DataFrame(zones)

    if zones_df.empty:
        return zones_df

    return zones_df.sort_values(
        ["strength_score", "reaction_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def candle_touches_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    return candle_high >= zone_low and candle_low <= zone_high


def find_first_zone_touch(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    zone_low: float,
    zone_high: float,
) -> int | None:
    for index in range(start_index, end_index + 1):
        if candle_touches_zone(
            float(df.at[index, "low"]),
            float(df.at[index, "high"]),
            zone_low,
            zone_high,
        ):
            return index

    return None

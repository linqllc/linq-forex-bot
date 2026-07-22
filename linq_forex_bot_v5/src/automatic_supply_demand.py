from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


Direction = Literal["long", "short"]
ZoneType = Literal["demand", "supply"]


@dataclass(frozen=True)
class AutomaticZone:
    zone_id: str
    instrument: str
    zone_type: ZoneType
    created_time: str
    origin_index: int
    impulse_end_index: int
    zone_top: float
    zone_bottom: float
    impulse_atr: float
    impulse_candles: int
    has_fvg: bool
    break_of_structure: bool
    touches: int
    fresh: bool
    invalidated: bool


@dataclass(frozen=True)
class AutomaticSetup:
    setup_id: str
    instrument: str
    timestamp: str
    direction: Direction
    zone_id: str
    zone_top: float
    zone_bottom: float
    entry: float
    stop: float
    risk_distance: float
    touches: int
    fresh_zone: bool
    has_fvg: bool
    break_of_structure: bool
    higher_timeframe_aligned: bool
    slow_pullback: bool
    confirmation_candle: bool
    discount_or_premium: bool
    zone_rejection: bool
    confluence_score: float
    grade: str


def _ensure_columns(candles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "time",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
    }

    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(
            "Candle data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = candles.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result = result.sort_values("time").reset_index(drop=True)

    numeric = ["mid_open", "mid_high", "mid_low", "mid_close"]
    result[numeric] = result[numeric].apply(
        pd.to_numeric,
        errors="coerce",
    )
    result = result.dropna(subset=numeric).reset_index(drop=True)

    return result


def add_detection_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_columns(candles)

    frame["body"] = (
        frame["mid_close"] - frame["mid_open"]
    ).abs()
    frame["range"] = frame["mid_high"] - frame["mid_low"]
    frame["bullish"] = frame["mid_close"] > frame["mid_open"]
    frame["bearish"] = frame["mid_close"] < frame["mid_open"]

    previous_close = frame["mid_close"].shift(1)
    true_range = pd.concat(
        [
            frame["mid_high"] - frame["mid_low"],
            (frame["mid_high"] - previous_close).abs(),
            (frame["mid_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    frame["atr"] = true_range.rolling(14).mean()
    frame["average_body"] = frame["body"].rolling(20).mean()

    for length in (20, 50, 200):
        frame[f"ema_{length}"] = frame["mid_close"].ewm(
            span=length,
            adjust=False,
        ).mean()

    swing_window = 5
    frame["swing_high"] = (
        frame["mid_high"]
        == frame["mid_high"].rolling(
            swing_window * 2 + 1,
            center=True,
        ).max()
    )
    frame["swing_low"] = (
        frame["mid_low"]
        == frame["mid_low"].rolling(
            swing_window * 2 + 1,
            center=True,
        ).min()
    )

    frame["previous_swing_high"] = (
        frame["mid_high"].where(frame["swing_high"]).ffill().shift(1)
    )
    frame["previous_swing_low"] = (
        frame["mid_low"].where(frame["swing_low"]).ffill().shift(1)
    )

    frame["bullish_bos"] = (
        frame["mid_close"] > frame["previous_swing_high"]
    )
    frame["bearish_bos"] = (
        frame["mid_close"] < frame["previous_swing_low"]
    )

    frame["bullish_fvg"] = (
        frame["mid_low"] > frame["mid_high"].shift(2)
    )
    frame["bearish_fvg"] = (
        frame["mid_high"] < frame["mid_low"].shift(2)
    )

    rolling_high = frame["mid_high"].rolling(100).max()
    rolling_low = frame["mid_low"].rolling(100).min()
    midpoint = rolling_low + (rolling_high - rolling_low) * 0.5

    frame["discount"] = frame["mid_close"] <= midpoint
    frame["premium"] = frame["mid_close"] >= midpoint

    frame["trend_bullish"] = (
        (frame["mid_close"] > frame["ema_200"])
        & (frame["ema_20"] > frame["ema_50"])
        & (frame["ema_50"] > frame["ema_200"])
    )

    frame["trend_bearish"] = (
        (frame["mid_close"] < frame["ema_200"])
        & (frame["ema_20"] < frame["ema_50"])
        & (frame["ema_50"] < frame["ema_200"])
    )

    return frame


def _origin_zone(
    frame: pd.DataFrame,
    impulse_start: int,
    zone_type: ZoneType,
    search_back: int = 8,
) -> tuple[int, float, float] | None:
    lower = max(0, impulse_start - search_back)

    for index in range(impulse_start - 1, lower - 1, -1):
        row = frame.iloc[index]

        opposite = (
            bool(row["bearish"])
            if zone_type == "demand"
            else bool(row["bullish"])
        )

        if not opposite:
            continue

        small_body = (
            pd.notna(row["average_body"])
            and row["body"] <= row["average_body"]
        )

        if small_body:
            top = float(row["mid_high"])
            bottom = float(row["mid_low"])
        else:
            top = float(max(row["mid_open"], row["mid_close"]))
            bottom = float(min(row["mid_open"], row["mid_close"]))

        return index, top, bottom

    return None


def _count_zone_touches(
    frame: pd.DataFrame,
    start_index: int,
    zone_top: float,
    zone_bottom: float,
    zone_type: ZoneType,
) -> tuple[int, bool]:
    touches = 0
    previously_inside = False
    invalidated = False

    for index in range(start_index, len(frame)):
        row = frame.iloc[index]

        inside = (
            float(row["mid_low"]) <= zone_top
            and float(row["mid_high"]) >= zone_bottom
        )

        if inside and not previously_inside:
            touches += 1

        previously_inside = inside

        if zone_type == "demand":
            if float(row["mid_close"]) < zone_bottom:
                invalidated = True
                break
        else:
            if float(row["mid_close"]) > zone_top:
                invalidated = True
                break

    return touches, invalidated


def detect_automatic_zones(
    candles: pd.DataFrame,
    instrument: str,
    body_multiplier: float = 1.2,
    impulse_atr_multiplier: float = 1.5,
    minimum_impulse_candles: int = 3,
    maximum_zone_touches: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_detection_features(candles)
    zones: list[AutomaticZone] = []

    index = 25

    while index < len(frame) - minimum_impulse_candles:
        row = frame.iloc[index]

        if pd.isna(row["atr"]) or pd.isna(row["average_body"]):
            index += 1
            continue

        bullish_sequence = True
        bearish_sequence = True
        impulse_end = index

        for test_index in range(
            index,
            min(index + 8, len(frame)),
        ):
            test_row = frame.iloc[test_index]

            is_big = (
                test_row["body"]
                >= test_row["average_body"] * body_multiplier
            )

            if not is_big:
                break

            if bool(test_row["bullish"]):
                bearish_sequence = False
            elif bool(test_row["bearish"]):
                bullish_sequence = False
            else:
                break

            impulse_end = test_index

        impulse_count = impulse_end - index + 1

        if impulse_count < minimum_impulse_candles:
            index += 1
            continue

        if not bullish_sequence and not bearish_sequence:
            index += 1
            continue

        total_move = abs(
            float(frame.iloc[impulse_end]["mid_close"])
            - float(frame.iloc[index]["mid_open"])
        )

        impulse_atr = total_move / max(float(row["atr"]), 1e-12)

        if impulse_atr < impulse_atr_multiplier:
            index += 1
            continue

        zone_type: ZoneType = (
            "demand" if bullish_sequence else "supply"
        )

        origin = _origin_zone(frame, index, zone_type)
        if origin is None:
            index += 1
            continue

        origin_index, zone_top, zone_bottom = origin

        fvg_slice = frame.iloc[
            index : impulse_end + 2
        ]

        has_fvg = bool(
            fvg_slice[
                "bullish_fvg"
                if zone_type == "demand"
                else "bearish_fvg"
            ].any()
        )

        bos = bool(
            fvg_slice[
                "bullish_bos"
                if zone_type == "demand"
                else "bearish_bos"
            ].any()
        )

        touches, invalidated = _count_zone_touches(
            frame,
            impulse_end + 1,
            zone_top,
            zone_bottom,
            zone_type,
        )

        zone_id = (
            f"{instrument}-"
            f"{zone_type}-"
            f"{frame.iloc[origin_index]['time'].isoformat()}"
        )

        zones.append(
            AutomaticZone(
                zone_id=zone_id,
                instrument=instrument,
                zone_type=zone_type,
                created_time=frame.iloc[
                    origin_index
                ]["time"].isoformat(),
                origin_index=origin_index,
                impulse_end_index=impulse_end,
                zone_top=zone_top,
                zone_bottom=zone_bottom,
                impulse_atr=impulse_atr,
                impulse_candles=impulse_count,
                has_fvg=has_fvg,
                break_of_structure=bos,
                touches=touches,
                fresh=touches <= maximum_zone_touches,
                invalidated=invalidated,
            )
        )

        index = impulse_end + 1

    return (
        frame,
        pd.DataFrame([asdict(zone) for zone in zones]),
    )


def _slow_pullback(
    frame: pd.DataFrame,
    entry_index: int,
    direction: Direction,
    lookback: int = 5,
) -> bool:
    start = max(0, entry_index - lookback)
    recent = frame.iloc[start:entry_index]

    if recent.empty:
        return False

    opposite = (
        recent[recent["bearish"]]
        if direction == "long"
        else recent[recent["bullish"]]
    )

    if opposite.empty:
        return True

    average_opposite_body = opposite["body"].mean()
    average_atr = recent["atr"].mean()

    if pd.isna(average_atr) or average_atr <= 0:
        return False

    return bool(average_opposite_body <= average_atr * 0.8)


def _grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C+"
    return "C"


def generate_automatic_setups(
    feature_frame: pd.DataFrame,
    zones: pd.DataFrame,
    confirmation_lookahead: int = 100,
) -> pd.DataFrame:
    setups: list[AutomaticSetup] = []

    if zones.empty:
        return pd.DataFrame()

    for _, zone in zones.iterrows():
        if bool(zone["invalidated"]):
            continue

        direction: Direction = (
            "long"
            if zone["zone_type"] == "demand"
            else "short"
        )

        start = int(zone["impulse_end_index"]) + 1
        end = min(
            len(feature_frame),
            start + confirmation_lookahead,
        )

        for index in range(start, end):
            row = feature_frame.iloc[index]

            entered_zone = (
                float(row["mid_low"]) <= float(zone["zone_top"])
                and float(row["mid_high"])
                >= float(zone["zone_bottom"])
            )

            if not entered_zone:
                continue

            if direction == "long":
                zone_invalid = (
                    float(row["mid_close"])
                    < float(zone["zone_bottom"])
                )
                confirmation = bool(row["bullish"])
                higher_timeframe_aligned = bool(
                    row["trend_bullish"]
                )
                discount_or_premium = bool(row["discount"])
                stop = (
                    float(zone["zone_bottom"])
                    - float(row["atr"]) * 0.3
                )
                rejection = (
                    float(row["mid_low"])
                    < float(row["mid_close"])
                    and float(row["mid_close"])
                    >= float(zone["zone_bottom"])
                )
            else:
                zone_invalid = (
                    float(row["mid_close"])
                    > float(zone["zone_top"])
                )
                confirmation = bool(row["bearish"])
                higher_timeframe_aligned = bool(
                    row["trend_bearish"]
                )
                discount_or_premium = bool(row["premium"])
                stop = (
                    float(zone["zone_top"])
                    + float(row["atr"]) * 0.3
                )
                rejection = (
                    float(row["mid_high"])
                    > float(row["mid_close"])
                    and float(row["mid_close"])
                    <= float(zone["zone_top"])
                )

            if zone_invalid:
                break

            if not confirmation:
                continue

            slow_pullback = _slow_pullback(
                feature_frame,
                index,
                direction,
            )

            entry = float(row["mid_close"])
            risk_distance = abs(entry - stop)

            if risk_distance <= 0:
                continue

            score = 0.0
            score += 20 if bool(zone["fresh"]) else 0
            score += 15 if bool(zone["has_fvg"]) else 0
            score += 15 if bool(
                zone["break_of_structure"]
            ) else 0
            score += 15 if higher_timeframe_aligned else 0
            score += 10 if slow_pullback else 0
            score += 10 if confirmation else 0
            score += 10 if discount_or_premium else 0
            score += 5 if rejection else 0

            setup_id = (
                f"{zone['zone_id']}-"
                f"{row['time'].isoformat()}"
            )

            setups.append(
                AutomaticSetup(
                    setup_id=setup_id,
                    instrument=str(zone["instrument"]),
                    timestamp=row["time"].isoformat(),
                    direction=direction,
                    zone_id=str(zone["zone_id"]),
                    zone_top=float(zone["zone_top"]),
                    zone_bottom=float(zone["zone_bottom"]),
                    entry=entry,
                    stop=stop,
                    risk_distance=risk_distance,
                    touches=int(zone["touches"]),
                    fresh_zone=bool(zone["fresh"]),
                    has_fvg=bool(zone["has_fvg"]),
                    break_of_structure=bool(
                        zone["break_of_structure"]
                    ),
                    higher_timeframe_aligned=(
                        higher_timeframe_aligned
                    ),
                    slow_pullback=slow_pullback,
                    confirmation_candle=confirmation,
                    discount_or_premium=discount_or_premium,
                    zone_rejection=rejection,
                    confluence_score=score,
                    grade=_grade(score),
                )
            )

            break

    return pd.DataFrame(
        [asdict(setup) for setup in setups]
    )

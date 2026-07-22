from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time
from typing import Literal
import math

import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class Zone:
    direction: Direction
    created_index: int
    breakout_index: int
    zone_low: float
    zone_high: float
    created_time: str
    touched: bool = False


@dataclass
class Trade:
    instrument: str
    direction: Direction
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    stop: float
    target: float
    exit_price: float
    result_r: float
    outcome: str
    zone_low: float
    zone_high: float
    breakout_body_multiple: float
    bars_to_retest: int

    def to_dict(self) -> dict:
        return asdict(self)


def _clock(value: str) -> time:
    hour, minute = [int(x) for x in value.split(":")]
    return time(hour, minute)


def _pip_size(instrument: str) -> float:
    return 0.01 if instrument.endswith("JPY") else 0.0001


def _find_origin(df: pd.DataFrame, breakout_idx: int, direction: Direction, lookback: int) -> int | None:
    start = max(0, breakout_idx - lookback)
    for idx in range(breakout_idx - 1, start - 1, -1):
        bullish = df.at[idx, "mid_close"] > df.at[idx, "mid_open"]
        if direction == "long" and not bullish:
            return idx
        if direction == "short" and bullish:
            return idx
    return None


def _entry_price(row: pd.Series, direction: Direction) -> float:
    if direction == "long":
        return float(row.get("ask_high", row["mid_high"]))
    return float(row.get("bid_low", row["mid_low"]))


def run_strategy(df: pd.DataFrame, instrument: str, config: dict) -> pd.DataFrame:
    """Run a conservative, completed-candle supply/demand backtest."""
    impulse = config["impulse"]
    zone_cfg = config["zone"]
    risk = config["risk"]
    costs = config["costs"]
    session_cfg = config["session"]

    entry_start = _clock(session_cfg["entry_start"])
    entry_end = _clock(session_cfg["entry_end"])
    max_daily = int(risk["max_trades_per_pair_per_day"])
    reward_risk = float(risk["reward_to_risk"])
    stop_buffer_atr = float(risk["stop_atr_buffer"])
    expiration = int(zone_cfg["expiration_candles"])
    search_back = int(zone_cfg["search_back_candles"])

    spread_pips = float(costs["default_spread_pips"].get(instrument, 1.0))
    slippage_pips = float(costs["slippage_pips"])
    execution_cost = (spread_pips + slippage_pips) * _pip_size(instrument)

    trades: list[Trade] = []
    daily_count: dict = {}
    active_zone: Zone | None = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        if pd.isna(row.get("opening_range_high")) or pd.isna(row.get("atr")):
            continue

        day = row["session_date"]
        daily_count.setdefault(day, 0)
        current_clock = row["clock"]

        if not (entry_start <= current_clock <= entry_end):
            continue
        if daily_count[day] >= max_daily:
            continue

        bullish_break = (
            row["mid_close"] > row["opening_range_high"]
            and row["body"] >= impulse["body_multiplier"] * row["median_body"]
            and row["body_fraction"] >= impulse["minimum_body_fraction"]
        )
        bearish_break = (
            row["mid_close"] < row["opening_range_low"]
            and row["body"] >= impulse["body_multiplier"] * row["median_body"]
            and row["body_fraction"] >= impulse["minimum_body_fraction"]
        )

        if active_zone is None and (bullish_break or bearish_break):
            direction: Direction = "long" if bullish_break else "short"
            origin_idx = _find_origin(df, i, direction, search_back)
            if origin_idx is not None:
                origin = df.iloc[origin_idx]
                displacement = abs(row["mid_close"] - origin["mid_close"])
                if displacement >= impulse["minimum_displacement_atr"] * row["atr"]:
                    active_zone = Zone(
                        direction=direction,
                        created_index=origin_idx,
                        breakout_index=i,
                        zone_low=float(origin["mid_low"]),
                        zone_high=float(origin["mid_high"]),
                        created_time=str(origin["time"]),
                    )
            continue

        if active_zone is None:
            continue

        bars_since_breakout = i - active_zone.breakout_index
        if bars_since_breakout > expiration or day != df.iloc[active_zone.breakout_index]["session_date"]:
            active_zone = None
            continue
        if i <= active_zone.breakout_index:
            continue

        touched = row["mid_low"] <= active_zone.zone_high and row["mid_high"] >= active_zone.zone_low
        if not touched:
            continue

        # Confirmation: reject from zone and close through the candle midpoint.
        candle_mid = (row["mid_high"] + row["mid_low"]) / 2
        if active_zone.direction == "long":
            confirmed = row["mid_close"] > candle_mid and row["mid_close"] > row["mid_open"]
            invalid = row["mid_close"] < active_zone.zone_low
        else:
            confirmed = row["mid_close"] < candle_mid and row["mid_close"] < row["mid_open"]
            invalid = row["mid_close"] > active_zone.zone_high

        if invalid:
            active_zone = None
            continue
        if not confirmed:
            active_zone.touched = True
            continue

        atr = float(row["atr"])
        if active_zone.direction == "long":
            entry = _entry_price(row, "long") + execution_cost / 2
            stop = active_zone.zone_low - stop_buffer_atr * atr
            risk_distance = entry - stop
            target = entry + reward_risk * risk_distance
        else:
            entry = _entry_price(row, "short") - execution_cost / 2
            stop = active_zone.zone_high + stop_buffer_atr * atr
            risk_distance = stop - entry
            target = entry - reward_risk * risk_distance

        if risk_distance <= 0 or not math.isfinite(risk_distance):
            active_zone = None
            continue

        exit_price = float(df.iloc[min(i + 1, len(df) - 1)]["mid_close"])
        exit_time = str(df.iloc[min(i + 1, len(df) - 1)]["time"])
        outcome = "open_at_data_end"
        result_r = (
            (exit_price - entry) / risk_distance
            if active_zone.direction == "long"
            else (entry - exit_price) / risk_distance
        )

        for j in range(i + 1, len(df)):
            future = df.iloc[j]
            if future["session_date"] != day:
                exit_price = float(future["mid_open"])
                exit_time = str(future["time"])
                outcome = "session_exit"
                result_r = (
                    (exit_price - entry) / risk_distance
                    if active_zone.direction == "long"
                    else (entry - exit_price) / risk_distance
                )
                break

            if active_zone.direction == "long":
                stop_hit = future["bid_low"] <= stop if "bid_low" in future else future["mid_low"] <= stop
                target_hit = future["bid_high"] >= target if "bid_high" in future else future["mid_high"] >= target
            else:
                stop_hit = future["ask_high"] >= stop if "ask_high" in future else future["mid_high"] >= stop
                target_hit = future["ask_low"] <= target if "ask_low" in future else future["mid_low"] <= target

            # Conservative assumption: if both occur inside the same candle, stop wins.
            if stop_hit:
                exit_price = stop
                exit_time = str(future["time"])
                outcome = "loss"
                result_r = -1.0
                break
            if target_hit:
                exit_price = target
                exit_time = str(future["time"])
                outcome = "win"
                result_r = reward_risk
                break

            if future["clock"] >= entry_end:
                exit_price = float(future["mid_close"])
                exit_time = str(future["time"])
                outcome = "time_exit"
                result_r = (
                    (exit_price - entry) / risk_distance
                    if active_zone.direction == "long"
                    else (entry - exit_price) / risk_distance
                )
                break

        body_multiple = float(row["body"] / row["median_body"]) if row["median_body"] else 0.0
        trades.append(
            Trade(
                instrument=instrument,
                direction=active_zone.direction,
                signal_time=str(row["time"]),
                entry_time=str(row["time"]),
                exit_time=exit_time,
                entry=entry,
                stop=stop,
                target=target,
                exit_price=exit_price,
                result_r=result_r,
                outcome=outcome,
                zone_low=active_zone.zone_low,
                zone_high=active_zone.zone_high,
                breakout_body_multiple=body_multiple,
                bars_to_retest=bars_since_breakout,
            )
        )
        daily_count[day] += 1
        active_zone = None

    return pd.DataFrame([trade.to_dict() for trade in trades])

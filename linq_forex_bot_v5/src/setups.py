from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import time
from typing import Literal, Callable
import math
import pandas as pd

Direction = Literal["long", "short"]

@dataclass
class Setup:
    instrument: str
    direction: Direction
    signal_index: int
    entry_index: int
    signal_time: str
    entry_time: str
    session_date: str
    entry: float
    stop: float
    risk_distance: float
    zone_low: float
    zone_high: float
    score: float
    trend_aligned: bool
    bos_confirmed: bool
    breakout_body_multiple: float
    displacement_atr: float
    bars_to_retest: int
    max_favorable_r: float
    stopped: bool
    final_r: float
    exit_time: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clock(value: str) -> time:
    h, m = map(int, value.split(":")); return time(h, m)


def _pip_size(instrument: str) -> float:
    return 0.01 if instrument.endswith("JPY") else 0.0001


def _find_origin(df: pd.DataFrame, breakout_idx: int, direction: Direction, lookback: int) -> int | None:
    for idx in range(breakout_idx - 1, max(-1, breakout_idx - lookback - 1), -1):
        bullish = df.at[idx, "mid_close"] > df.at[idx, "mid_open"]
        if direction == "long" and not bullish: return idx
        if direction == "short" and bullish: return idx
    return None


def _quality_score(row: pd.Series, direction: Direction, body_multiple: float, displacement_atr: float, fresh: bool, config: dict) -> tuple[float,bool,bool]:
    w = config["quality"]["weights"]
    trend_aligned = (direction == "long" and row["trend"] == "bullish") or (direction == "short" and row["trend"] == "bearish")
    bos = bool(row["bullish_bos"] if direction == "long" else row["bearish_bos"])

    if direction == "long":
        mtf_alignment = float(row.get("mtf_bullish_alignment", 0.0))
    else:
        mtf_alignment = float(row.get("mtf_bearish_alignment", 0.0))

    mtf_aligned = mtf_alignment >= float(
        config.get("multi_timeframe", {}).get("minimum_alignment", 1.0)
    )

    score = 0.0
    score += w["trend_alignment"] if trend_aligned else 0
    score += w.get("multi_timeframe_alignment", 0) * mtf_alignment
    score += w["break_of_structure"] if bos else 0
    score += w["fresh_zone"] if fresh else 0
    score += w["impulse_strength"] * min(1.0, body_multiple / max(config["impulse"]["body_multiplier"] * 1.75, 1e-9))
    score += w["displacement"] * min(1.0, displacement_atr / max(config["impulse"]["minimum_displacement_atr"] * 1.75, 1e-9))
    score += w["opening_range_break"]
    return round(score, 2), trend_aligned, bos


def scan_setups(df: pd.DataFrame, instrument: str, config: dict, progress_callback: Callable[[int, int, str], None] | None = None) -> pd.DataFrame:
    """Scan once. Store each setup's MFE and stop outcome so every R target can be evaluated quickly."""
    entry_start, entry_end = _clock(config["session"]["entry_start"]), _clock(config["session"]["entry_end"])
    impulse, zone_cfg, risk_cfg = config["impulse"], config["zone"], config["risk"]
    spread = config["costs"]["default_spread_pips"].get(instrument, 1.0)
    cost = (spread + config["costs"]["slippage_pips"]) * _pip_size(instrument)
    setups, daily_count = [], {}
    active = None

    total_rows = max(1, len(df) - 1)
    for i in range(1, len(df)):
        if progress_callback and (i % 500 == 0 or i == len(df) - 1):
            progress_callback(i, total_rows, f"{len(setups)} setups found")
        row = df.iloc[i]
        if pd.isna(row.get("atr")) or pd.isna(row.get("opening_range_high")): continue
        day, clock = row["session_date"], row["clock"]
        daily_count.setdefault(day, 0)
        if not (entry_start <= clock <= entry_end) or daily_count[day] >= risk_cfg["max_trades_per_pair_per_day"]: continue

        bullish_break = row["mid_close"] > row["opening_range_high"] and row["body"] >= impulse["body_multiplier"] * row["median_body"] and row["body_fraction"] >= impulse["minimum_body_fraction"]
        bearish_break = row["mid_close"] < row["opening_range_low"] and row["body"] >= impulse["body_multiplier"] * row["median_body"] and row["body_fraction"] >= impulse["minimum_body_fraction"]
        if active is None and (bullish_break or bearish_break):
            direction: Direction = "long" if bullish_break else "short"

            mtf_cfg = config.get("multi_timeframe", {})
            if mtf_cfg.get("enabled", False):
                alignment_column = (
                    "mtf_bullish_alignment"
                    if direction == "long"
                    else "mtf_bearish_alignment"
                )
                alignment = float(row.get(alignment_column, 0.0))
                required = float(mtf_cfg.get("minimum_alignment", 1.0))

                if mtf_cfg.get("require_alignment", True) and alignment < required:
                    continue

            origin_idx = _find_origin(df, i, direction, zone_cfg["search_back_candles"])
            if origin_idx is None: continue
            origin = df.iloc[origin_idx]
            displacement_atr = abs(row["mid_close"] - origin["mid_close"]) / max(row["atr"], 1e-12)
            width_atr = (origin["mid_high"] - origin["mid_low"]) / max(row["atr"], 1e-12)
            if displacement_atr < impulse["minimum_displacement_atr"] or width_atr > zone_cfg["maximum_width_atr"]: continue
            body_multiple = row["body"] / max(row["median_body"], 1e-12)
            score, aligned, bos = _quality_score(row, direction, body_multiple, displacement_atr, True, config)
            active = dict(direction=direction, origin_idx=origin_idx, breakout_idx=i, day=day,
                          low=float(origin["mid_low"]), high=float(origin["mid_high"]), score=score,
                          aligned=aligned, bos=bos, body_multiple=body_multiple, displacement_atr=displacement_atr)
            continue
        if active is None: continue
        if day != active["day"] or i - active["breakout_idx"] > zone_cfg["expiration_candles"]:
            active = None; continue
        touched = row["mid_low"] <= active["high"] and row["mid_high"] >= active["low"]
        if not touched: continue
        midpoint = (row["mid_high"] + row["mid_low"]) / 2
        if active["direction"] == "long":
            invalid = row["mid_close"] < active["low"]
            confirmed = row["mid_close"] > midpoint and row["mid_close"] > row["mid_open"]
            entry = float(row.get("ask_high", row["mid_high"])) + cost / 2
            stop = active["low"] - risk_cfg["stop_atr_buffer"] * row["atr"]
            risk_distance = entry - stop
        else:
            invalid = row["mid_close"] > active["high"]
            confirmed = row["mid_close"] < midpoint and row["mid_close"] < row["mid_open"]
            entry = float(row.get("bid_low", row["mid_low"])) - cost / 2
            stop = active["high"] + risk_cfg["stop_atr_buffer"] * row["atr"]
            risk_distance = stop - entry
        if invalid: active = None; continue
        if not confirmed: continue
        if active["score"] < config["quality"]["minimum_score"] or risk_distance <= 0 or not math.isfinite(risk_distance):
            active = None; continue

        max_favorable_r, stopped, final_r = 0.0, False, 0.0
        exit_time = str(row["time"])
        for j in range(i + 1, len(df)):
            future = df.iloc[j]
            if future["session_date"] != day or future["clock"] >= entry_end:
                px = float(future["mid_open"] if future["session_date"] != day else future["mid_close"])
                final_r = (px-entry)/risk_distance if active["direction"] == "long" else (entry-px)/risk_distance
                exit_time = str(future["time"]); break
            if active["direction"] == "long":
                stop_hit = float(future.get("bid_low", future["mid_low"])) <= stop
                favorable = (float(future.get("bid_high", future["mid_high"])) - entry) / risk_distance
            else:
                stop_hit = float(future.get("ask_high", future["mid_high"])) >= stop
                favorable = (entry - float(future.get("ask_low", future["mid_low"]))) / risk_distance
            max_favorable_r = max(max_favorable_r, favorable)
            if stop_hit:
                stopped, final_r, exit_time = True, -1.0, str(future["time"]); break
        setups.append(Setup(instrument, active["direction"], active["breakout_idx"], i, str(df.iloc[active["breakout_idx"]]["time"]), str(row["time"]), str(day), entry, stop, risk_distance, active["low"], active["high"], active["score"], active["aligned"], active["bos"], active["body_multiple"], active["displacement_atr"], i-active["breakout_idx"], max_favorable_r, stopped, final_r, exit_time).to_dict())
        daily_count[day] += 1
        active = None
    if progress_callback:
        progress_callback(total_rows, total_rows, f"{len(setups)} setups found")
    return pd.DataFrame(setups)

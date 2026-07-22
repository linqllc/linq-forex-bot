from __future__ import annotations

import math
import pandas as pd
from src.engine_features import build_engine_features


def _safe_div(a: float, b: float) -> float:
    if b is None or not math.isfinite(float(b)) or abs(float(b)) < 1e-12:
        return 0.0
    return float(a) / float(b)


def build_feature_dataset(setups: pd.DataFrame, candles: pd.DataFrame, ratios: list[float]) -> pd.DataFrame:
    """Create one ML-ready row per setup, with only information known at entry plus outcome labels."""
    if setups.empty:
        cols = [
            "instrument", "entry_time", "session_date", "direction", "score",
            "trend_aligned", "bos_confirmed", "breakout_body_multiple",
            "displacement_atr", "bars_to_retest", "zone_width_atr",
            "entry_atr", "ema_distance_atr", "opening_range_position",
            "hour_utc", "weekday",
            "h1_trend", "h4_trend", "h1_h4_agree",
            "mtf_direction_alignment",
            "max_favorable_r", "stopped", "final_r",
        ]
        cols += [f"hit_{r:.2f}R" for r in ratios]
        return pd.DataFrame(columns=cols)

    candle_lookup = candles.reset_index(drop=True)
    rows: list[dict] = []
    for setup in setups.to_dict("records"):
        idx = int(setup["entry_index"])
        if idx < 0 or idx >= len(candle_lookup):
            continue
        c = candle_lookup.iloc[idx]
        atr = float(c.get("atr", 0.0) or 0.0)
        entry = float(setup["entry"])
        zone_width = float(setup["zone_high"]) - float(setup["zone_low"])
        ema_fast = float(c.get("ema_fast", c.get("mid_close", entry)))
        or_low = float(c.get("opening_range_low", entry))
        or_high = float(c.get("opening_range_high", entry))
        or_span = max(or_high - or_low, 1e-12)
        timestamp = pd.Timestamp(setup["entry_time"])
        row = {
            "instrument": setup["instrument"],
            "entry_time": setup["entry_time"],
            "session_date": setup["session_date"],
            "direction": setup["direction"],
            "score": float(setup["score"]),
            "trend_aligned": int(bool(setup["trend_aligned"])),
            "bos_confirmed": int(bool(setup["bos_confirmed"])),
            "breakout_body_multiple": float(setup["breakout_body_multiple"]),
            "displacement_atr": float(setup["displacement_atr"]),
            "bars_to_retest": int(setup["bars_to_retest"]),
            "zone_width_atr": _safe_div(zone_width, atr),
            "entry_atr": atr,
            "ema_distance_atr": _safe_div(abs(entry - ema_fast), atr),
            "opening_range_position": (entry - or_low) / or_span,
            "hour_utc": int(timestamp.hour),
            "weekday": int(timestamp.dayofweek),
            "h1_trend": str(c.get("h1_trend", "neutral")),
            "h4_trend": str(c.get("h4_trend", "neutral")),
            "h1_h4_agree": int(c.get("h1_h4_agree", 0)),
            "mtf_direction_alignment": float(
                c.get(
                    "mtf_bullish_alignment"
                    if setup["direction"] == "long"
                    else "mtf_bearish_alignment",
                    0.0,
                )
            ),
            "max_favorable_r": float(setup["max_favorable_r"]),
            "stopped": int(bool(setup["stopped"])),
            "final_r": float(setup["final_r"]),
        }
        row.update(build_engine_features(setup, c, candle_lookup))
        for ratio in ratios:
            row[f"hit_{ratio:.2f}R"] = int(float(setup["max_favorable_r"]) >= float(ratio))
        rows.append(row)
    return pd.DataFrame(rows)

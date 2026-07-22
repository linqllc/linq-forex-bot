from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(df: pd.DataFrame, atr_period: int, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    out = df.copy()

    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["tr"] = true_range
    out["atr"] = true_range.ewm(alpha=1 / atr_period, adjust=False).mean()
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["ema_slow_slope_12"] = out["ema_slow"].diff(12) / out["atr"].replace(0, np.nan)

    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_ratio"] = out["body"] / candle_range
    out["range_atr"] = candle_range / out["atr"].replace(0, np.nan)
    out["close_location"] = (out["close"] - out["low"]) / candle_range
    out["ema_distance_atr"] = (out["close"] - out["ema_slow"]) / out["atr"].replace(0, np.nan)
    out["return_12"] = out["close"].pct_change(12)
    out["return_48"] = out["close"].pct_change(48)
    out["realized_vol_48"] = out["close"].pct_change().rolling(48).std()

    ts = out["timestamp"]
    out["hour_utc"] = ts.dt.hour + ts.dt.minute / 60.0
    out["weekday"] = ts.dt.weekday
    out["session"] = np.select(
        [
            (out["hour_utc"] >= 6) & (out["hour_utc"] < 8),
            (out["hour_utc"] >= 8) & (out["hour_utc"] < 12),
            (out["hour_utc"] >= 12) & (out["hour_utc"] < 16),
        ],
        ["london_open", "london", "new_york_overlap"],
        default="off_session",
    )

    rolling_high = out["high"].shift(1).rolling(48).max()
    rolling_low = out["low"].shift(1).rolling(48).min()
    out["distance_prev_high_atr"] = (rolling_high - out["close"]) / out["atr"].replace(0, np.nan)
    out["distance_prev_low_atr"] = (out["close"] - rolling_low) / out["atr"].replace(0, np.nan)

    # Three-candle fair value gap magnitude.
    bull_gap = out["low"] - out["high"].shift(2)
    bear_gap = out["low"].shift(2) - out["high"]
    out["bull_fvg_atr"] = bull_gap.clip(lower=0) / out["atr"].replace(0, np.nan)
    out["bear_fvg_atr"] = bear_gap.clip(lower=0) / out["atr"].replace(0, np.nan)

    return out

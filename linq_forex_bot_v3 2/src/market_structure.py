from __future__ import annotations
import numpy as np
import pandas as pd


def add_market_structure(df: pd.DataFrame, fast_ema: int = 20, slow_ema: int = 50, swing_window: int = 3) -> pd.DataFrame:
    """Completed-candle EMA trend, confirmed swings, BOS and CHoCH-like flags."""
    out = df.copy()
    close = out["mid_close"]
    out["ema_fast"] = close.ewm(span=fast_ema, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=slow_ema, adjust=False).mean()
    out["trend"] = np.select(
        [out["ema_fast"] > out["ema_slow"], out["ema_fast"] < out["ema_slow"]],
        ["bullish", "bearish"], default="neutral"
    )
    width = swing_window * 2 + 1
    raw_high = out["mid_high"].where(out["mid_high"].eq(out["mid_high"].rolling(width, center=True).max()))
    raw_low = out["mid_low"].where(out["mid_low"].eq(out["mid_low"].rolling(width, center=True).min()))
    out["confirmed_swing_high"] = raw_high.shift(swing_window)
    out["confirmed_swing_low"] = raw_low.shift(swing_window)
    out["last_swing_high"] = out["confirmed_swing_high"].ffill().shift(1)
    out["last_swing_low"] = out["confirmed_swing_low"].ffill().shift(1)
    out["bullish_bos"] = (close > out["last_swing_high"]) & (close.shift(1) <= out["last_swing_high"].shift(1))
    out["bearish_bos"] = (close < out["last_swing_low"]) & (close.shift(1) >= out["last_swing_low"].shift(1))
    previous_trend = out["trend"].shift(1)
    out["bullish_choch"] = out["bullish_bos"] & previous_trend.eq("bearish")
    out["bearish_choch"] = out["bearish_bos"] & previous_trend.eq("bullish")
    return out

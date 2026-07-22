from __future__ import annotations
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame, atr_period: int = 14, body_lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    high = out["mid_high"]
    low = out["mid_low"]
    close = out["mid_close"]
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr"] = true_range.rolling(atr_period, min_periods=atr_period).mean()
    out["body"] = (out["mid_close"] - out["mid_open"]).abs()
    out["range"] = (out["mid_high"] - out["mid_low"]).replace(0, np.nan)
    out["body_fraction"] = out["body"] / out["range"]
    out["median_body"] = out["body"].shift(1).rolling(
        body_lookback, min_periods=body_lookback
    ).median()
    return out

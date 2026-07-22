from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def generate_demo_data(days: int = 90, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic M5 candles to verify the software pipeline only."""
    rng = np.random.default_rng(seed)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    periods = days * 24 * 12
    times = pd.date_range(start, periods=periods, freq="5min", tz="UTC")

    returns = rng.normal(0, 0.00018, periods)
    close = 1.10 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    wiggle = np.abs(rng.normal(0.00012, 0.00006, periods))
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    spread = 0.00008

    frame = pd.DataFrame(
        {
            "time": times,
            "volume": rng.integers(50, 500, periods),
            "complete": True,
            "mid_open": open_,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
            "bid_open": open_ - spread / 2,
            "bid_high": high - spread / 2,
            "bid_low": low - spread / 2,
            "bid_close": close - spread / 2,
            "ask_open": open_ + spread / 2,
            "ask_high": high + spread / 2,
            "ask_low": low + spread / 2,
            "ask_close": close + spread / 2,
        }
    )
    return frame

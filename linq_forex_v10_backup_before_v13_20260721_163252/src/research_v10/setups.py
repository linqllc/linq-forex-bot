from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


def detect_candidate_setups(df: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    """
    Detects objective impulse/pullback candidates without requiring legacy V7 zones.

    Long:
      bullish displacement candle -> retrace 25-75% of impulse within N bars.
    Short:
      bearish displacement candle -> retrace 25-75% of impulse within N bars.
    """
    rows: list[dict] = []
    warmup = max(cfg.ema_slow + 12, 250)

    for i in range(warmup, len(df) - cfg.max_holding_bars - 1):
        candle = df.iloc[i]
        if not np.isfinite(candle["atr"]) or candle["atr"] <= 0:
            continue

        bullish = candle["close"] > candle["open"]
        bearish = candle["close"] < candle["open"]
        displaced = (
            candle["range_atr"] >= cfg.displacement_atr_min
            and candle["body_ratio"] >= cfg.displacement_body_ratio_min
        )
        if not displaced or not (bullish or bearish):
            continue

        direction = "long" if bullish else "short"
        impulse_low = float(candle["low"])
        impulse_high = float(candle["high"])
        impulse_size = impulse_high - impulse_low
        if impulse_size <= 0:
            continue

        if direction == "long":
            shallow = impulse_high - cfg.pullback_min * impulse_size
            deep = impulse_high - cfg.pullback_max * impulse_size
        else:
            shallow = impulse_low + cfg.pullback_min * impulse_size
            deep = impulse_low + cfg.pullback_max * impulse_size

        entry_idx = None
        entry_price = None
        retrace_fraction = None

        end = min(i + cfg.max_pullback_bars + 1, len(df))
        for j in range(i + 1, end):
            future = df.iloc[j]
            if direction == "long":
                # Invalidation before entry: full impulse low broken.
                if future["low"] < impulse_low:
                    break
                touched = future["low"] <= shallow and future["high"] >= deep
                if touched:
                    entry_price = min(shallow, max(deep, float(future["close"])))
                    retrace_fraction = (impulse_high - entry_price) / impulse_size
                    entry_idx = j
                    break
            else:
                if future["high"] > impulse_high:
                    break
                touched = future["high"] >= shallow and future["low"] <= deep
                if touched:
                    entry_price = max(shallow, min(deep, float(future["close"])))
                    retrace_fraction = (entry_price - impulse_low) / impulse_size
                    entry_idx = j
                    break

        if entry_idx is None or entry_price is None:
            continue

        entry = df.iloc[entry_idx]
        rows.append(
            {
                "setup_id": f"{cfg.instrument}-{direction}-{entry['timestamp'].isoformat()}",
                "instrument": cfg.instrument,
                "direction": direction,
                "impulse_index": i,
                "entry_index": entry_idx,
                "impulse_timestamp": candle["timestamp"],
                "timestamp": entry["timestamp"],
                "entry_price": entry_price,
                "impulse_atr": impulse_size / candle["atr"],
                "impulse_body_ratio": candle["body_ratio"],
                "retrace_fraction": retrace_fraction,
                "pullback_bars": entry_idx - i,
                "atr": entry["atr"],
                "ema_fast": entry["ema_fast"],
                "ema_slow": entry["ema_slow"],
                "ema_slow_slope_12": entry["ema_slow_slope_12"],
                "ema_distance_atr": entry["ema_distance_atr"],
                "return_12": entry["return_12"],
                "return_48": entry["return_48"],
                "realized_vol_48": entry["realized_vol_48"],
                "hour_utc": entry["hour_utc"],
                "weekday": entry["weekday"],
                "session": entry["session"],
                "distance_prev_high_atr": entry["distance_prev_high_atr"],
                "distance_prev_low_atr": entry["distance_prev_low_atr"],
                "bull_fvg_atr": candle["bull_fvg_atr"],
                "bear_fvg_atr": candle["bear_fvg_atr"],
                "trend_aligned": (
                    entry_price > entry["ema_slow"] and entry["ema_slow_slope_12"] > 0
                    if direction == "long"
                    else entry_price < entry["ema_slow"] and entry["ema_slow_slope_12"] < 0
                ),
            }
        )

    setups = pd.DataFrame(rows)
    if setups.empty:
        return setups

    # Prevent overlapping duplicate impulses from generating multiple simultaneous trades.
    setups = (
        setups.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp", "direction"], keep="first")
        .reset_index(drop=True)
    )
    return setups

from __future__ import annotations

from datetime import time
import math
import pandas as pd


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _safe(value, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _session_score(clock_value: time) -> float:
    # New York morning liquidity window. This is a measurable heuristic, not a claim of edge.
    minutes = clock_value.hour * 60 + clock_value.minute
    if 9 * 60 + 45 <= minutes <= 10 * 60 + 30:
        return 100.0
    if 10 * 60 + 30 < minutes <= 11 * 60 + 15:
        return 80.0
    if 11 * 60 + 15 < minutes <= 12 * 60:
        return 60.0
    return 25.0


def build_engine_features(setup: dict, candle: pd.Series, candles: pd.DataFrame) -> dict:
    """Generate transparent engine scores using entry-time information only."""
    idx = int(setup["entry_index"])
    atr = max(_safe(candle.get("atr")), 1e-12)
    ema_fast = _safe(candle.get("ema_fast"), _safe(candle.get("mid_close")))
    ema_slow = _safe(candle.get("ema_slow"), ema_fast)
    close = _safe(candle.get("mid_close"), _safe(setup.get("entry")))

    trend_aligned = bool(setup.get("trend_aligned"))
    ema_separation_atr = abs(ema_fast - ema_slow) / atr
    trend_score = _clip((55.0 if trend_aligned else 20.0) + min(45.0, ema_separation_atr * 75.0))

    bos = bool(setup.get("bos_confirmed"))
    direction = setup.get("direction")
    choch = bool(candle.get("bullish_choch", False) if direction == "long" else candle.get("bearish_choch", False))
    structure_score = _clip((65.0 if bos else 30.0) + (20.0 if choch else 0.0) + min(15.0, _safe(setup.get("displacement_atr")) * 5.0))

    zone_width_atr = (_safe(setup.get("zone_high")) - _safe(setup.get("zone_low"))) / atr
    displacement = _safe(setup.get("displacement_atr"))
    retest_penalty = min(30.0, max(0, int(setup.get("bars_to_retest", 0)) - 1) * 2.0)
    supply_demand_score = _clip(45.0 + min(35.0, displacement * 12.0) + max(0.0, 20.0 - zone_width_atr * 12.0) - retest_penalty)

    atr_history = candles.loc[max(0, idx - 100): max(0, idx - 1), "atr"].dropna()
    median_atr = float(atr_history.median()) if not atr_history.empty else atr
    volatility_ratio = atr / max(median_atr, 1e-12)
    volatility_score = _clip(100.0 - abs(volatility_ratio - 1.25) * 70.0)

    lookback = candles.iloc[max(0, idx - 12): idx + 1]
    last_swing_low = _safe(candle.get("last_swing_low"), close)
    last_swing_high = _safe(candle.get("last_swing_high"), close)
    lows = lookback["mid_low"] if "mid_low" in lookback else pd.Series([close] * len(lookback), index=lookback.index)
    highs = lookback["mid_high"] if "mid_high" in lookback else pd.Series([close] * len(lookback), index=lookback.index)
    if direction == "long":
        liquidity_sweep = bool((lows < last_swing_low).any() and close > last_swing_low)
    else:
        liquidity_sweep = bool((highs > last_swing_high).any() and close < last_swing_high)
    liquidity_score = 90.0 if liquidity_sweep else 35.0

    clock_value = candle.get("clock")
    if not isinstance(clock_value, time):
        clock_value = time(10, 0)
    session_score = _session_score(clock_value)

    component_scores = {
        "trend_engine_score": round(trend_score, 2),
        "structure_engine_score": round(structure_score, 2),
        "supply_demand_engine_score": round(supply_demand_score, 2),
        "liquidity_engine_score": round(liquidity_score, 2),
        "volatility_engine_score": round(volatility_score, 2),
        "session_engine_score": round(session_score, 2),
    }
    weights = {
        "trend_engine_score": 0.20,
        "structure_engine_score": 0.20,
        "supply_demand_engine_score": 0.25,
        "liquidity_engine_score": 0.10,
        "volatility_engine_score": 0.15,
        "session_engine_score": 0.10,
    }
    confluence = sum(component_scores[k] * weights[k] for k in weights)
    return {
        **component_scores,
        "confluence_score": round(confluence, 2),
        "ema_separation_atr": round(ema_separation_atr, 6),
        "volatility_ratio": round(volatility_ratio, 6),
        "liquidity_sweep": int(liquidity_sweep),
        "choch_confirmed": int(choch),
    }

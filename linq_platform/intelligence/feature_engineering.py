"""
Native Phase 4 feature engineering.

This module contains the exact validated dataset-construction logic extracted
from the Phase 4 reference engine. It creates model-safe features, applies
spread and stop filters, labels historical outcomes, and prevents target
leakage.

Behavior is protected by direct native-versus-reference parity tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LEAKAGE = {
    "timestamp",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_price",
    "mfe_r",
    "mae_r",
    "stop_hit",
    "valid_risk",
    "bars_observed",
    "result_r",
    "gross_result_r",
    "net_result_r",
    "exit_reason",
    "bars_held",
    "exit_price",
    "spread_cost_r",
    "slippage_cost_r",
    "total_cost_r",
    "selected",
    "probability_1r",
    "actual_win",
    "holdout",
}


__all__ = [
    "LEAKAGE",
    "clean_features",
    "evaluate",
    "build_dataset",
    "feature_columns",
]


def clean_features(row):
    out = {}
    for k, v in row.items():
        s = str(k)
        if (
            k in LEAKAGE
            or s.startswith("hit_")
            or s.startswith("bars_to_")
            or s.startswith("probability")
        ):
            continue
        out[k] = v
    return out


def evaluate(direction, entry, stop, future, spread_pips, cfg):
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0 or future.empty:
        return None
    target = entry + risk if direction == "long" else entry - risk
    if direction == "long":
        target_hits = future["high"] >= target
        stop_hits = future["low"] <= stop
        fav = future["high"] - entry
        adv = entry - future["low"]
    else:
        target_hits = future["low"] <= target
        stop_hits = future["high"] >= stop
        fav = entry - future["low"]
        adv = future["high"] - entry
    ti = np.flatnonzero(target_hits.to_numpy())
    si = np.flatnonzero(stop_hits.to_numpy())
    ft = int(ti[0]) if len(ti) else None
    fs = int(si[0]) if len(si) else None
    won = ft is not None and (fs is None or ft < fs)
    if won:
        gross, bar, reason, exit_price = cfg.target_r, ft, "target", target
    elif fs is not None:
        gross, bar, reason, exit_price = -1.0, fs, "stop", stop
    else:
        last = float(future["close"].iloc[-1])
        signed = last - entry if direction == "long" else entry - last
        gross, bar, reason, exit_price = (
            float(np.clip(signed / risk, -1, cfg.target_r)),
            len(future) - 1,
            "time_exit",
            last,
        )
    spread = float(spread_pips) if np.isfinite(spread_pips) else 0.0
    spread_r = spread * cfg.pip_size / risk
    slip_r = 2 * cfg.slippage_pips_each_side * cfg.pip_size / risk
    cost = spread_r + slip_r
    return {
        "actual_win": int(won),
        "gross_result_r": gross,
        "net_result_r": gross - cost,
        "exit_reason": reason,
        "bars_held": bar + 1,
        "exit_price": exit_price,
        "target_price": target,
        "risk_price": risk,
        "mfe_r": float((fav / risk).max()),
        "mae_r": float((adv / risk).max()),
        "spread_cost_r": spread_r,
        "slippage_cost_r": slip_r,
        "total_cost_r": cost,
    }


def build_dataset(candles, setups, phase1, cfg):
    fmap = phase1.set_index("setup_id", drop=False).to_dict("index")
    times = candles["timestamp"]
    rows = []
    for _, setup in setups.iterrows():
        pos = int(times.searchsorted(setup["timestamp"], side="left"))
        if pos >= len(candles) or pos < 200:
            continue
        candle = candles.iloc[pos]
        atr = float(candle["atr"]) if pd.notna(candle["atr"]) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue
        spread = candle.get("spread_pips", np.nan)

        if (
            cfg.maximum_spread_pips is not None
            and np.isfinite(spread)
            and spread > cfg.maximum_spread_pips
        ):
            continue
        future = candles.iloc[pos + 1 : pos + 1 + cfg.forward_bars]
        if future.empty:
            continue
        entry = (
            float(setup["entry"])
            if "entry" in setup.index and pd.notna(setup["entry"])
            else float(candle["close"])
        )
        atr_stop = cfg.stop_atr * atr
        minimum_stop = cfg.minimum_stop_pips * cfg.pip_size
        stop_distance = max(atr_stop, minimum_stop)

        stop = entry - stop_distance if setup["direction"] == "long" else entry + stop_distance
        stop_distance_pips = abs(entry - stop) / cfg.pip_size

        if (
            np.isfinite(spread)
            and stop_distance_pips > 0
            and spread / stop_distance_pips > cfg.maximum_spread_to_stop_ratio
        ):
            continue

        outcome = evaluate(
            setup["direction"],
            entry,
            stop,
            future,
            spread,
            cfg,
        )
        if outcome is None:
            continue
        row = clean_features(fmap.get(setup["setup_id"], {}))
        row.update(
            {
                "setup_id": setup["setup_id"],
                "timestamp": setup["timestamp"],
                "direction": setup["direction"],
                "entry_price": entry,
                "stop_price": stop,
                "stop_distance_atr": cfg.stop_atr,
                "spread_pips_at_entry": spread,
                "atr_at_entry": atr,
                "market_ema20_50_atr": (candle["ema20"] - candle["ema50"]) / atr,
                "market_ema50_200_atr": (candle["ema50"] - candle["ema200"]) / atr,
                "market_rsi": candle["rsi"],
                "market_ret3": candle["ret3"],
                "market_ret12": candle["ret12"],
                "market_range_atr": candle["range_atr"],
                "market_hour_utc": int(candle["hour_utc"]),
                "market_weekday": int(candle["weekday"]),
            }
        )
        row.update(outcome)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def feature_columns(df):
    excluded = LEAKAGE | {"setup_id", "strategy"}
    return [c for c in df.columns if c not in excluded]

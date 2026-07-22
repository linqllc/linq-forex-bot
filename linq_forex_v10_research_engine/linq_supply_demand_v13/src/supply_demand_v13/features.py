from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Config


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _confirmed_pivots(series: pd.Series, left: int, right: int, high: bool) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.to_numpy()
    for i in range(left, len(values) - right):
        window = values[i-left:i+right+1]
        center = values[i]
        valid = center == (np.max(window) if high else np.min(window))
        if valid:
            # Pivot becomes knowable only right bars later.
            out.iloc[i + right] = center
    return out


def add_features(m5: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = m5.copy()
    df["atr"] = _atr(df, cfg.atr_period)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).clip(lower=1e-12)
    df["body_range"] = df["body"] / df["range"]

    # Completed H1 bars only: right-closed/right-labeled.
    h1 = (
        df.set_index("time")
        .resample("1h", label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    h1["ema_fast"] = h1["close"].ewm(span=cfg.h1_ema_fast, adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=cfg.h1_ema_slow, adjust=False).mean()
    h1["pivot_high_known"] = _confirmed_pivots(
        h1["high"], cfg.h1_pivot_left, cfg.h1_pivot_right, True
    )
    h1["pivot_low_known"] = _confirmed_pivots(
        h1["low"], cfg.h1_pivot_left, cfg.h1_pivot_right, False
    )
    h1["last_high"] = h1["pivot_high_known"].ffill()
    h1["prev_high"] = h1["pivot_high_known"].where(
        h1["pivot_high_known"].notna()
    ).ffill().shift(1).ffill()
    h1["last_low"] = h1["pivot_low_known"].ffill()
    h1["prev_low"] = h1["pivot_low_known"].where(
        h1["pivot_low_known"].notna()
    ).ffill().shift(1).ffill()

    h1["bull_structure"] = (
        (h1["last_high"] > h1["prev_high"]) &
        (h1["last_low"] > h1["prev_low"])
    )
    h1["bear_structure"] = (
        (h1["last_high"] < h1["prev_high"]) &
        (h1["last_low"] < h1["prev_low"])
    )
    h1["h1_direction"] = np.select(
        [
            h1["bull_structure"] & (h1["ema_fast"] > h1["ema_slow"]),
            h1["bear_structure"] & (h1["ema_fast"] < h1["ema_slow"]),
        ],
        [1, -1],
        default=0,
    )

    context = h1[["time", "h1_direction", "last_high", "last_low"]].copy()
    df = pd.merge_asof(
        df.sort_values("time"),
        context.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )
    df["h1_direction"] = df["h1_direction"].fillna(0).astype(int)
    return df

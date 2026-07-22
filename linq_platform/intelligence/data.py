from __future__ import annotations

import numpy as np
import pandas as pd


def find_col(df, aliases, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(f"Missing one of {aliases}. Available: {list(df.columns)}")
    return None


def load_candles(path):
    raw = pd.read_csv(path)
    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "date"],
        "open": ["open", "o", "mid_open", "mid_o"],
        "high": ["high", "h", "mid_high", "mid_h"],
        "low": ["low", "l", "mid_low", "mid_l"],
        "close": ["close", "c", "mid_close", "mid_c"],
        "bid_close": ["bid_close", "bid_c"],
        "ask_close": ["ask_close", "ask_c"],
    }
    rename = {}
    for canonical, choices in aliases.items():
        col = find_col(raw, choices, canonical in {"timestamp", "open", "high", "low", "close"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in [x for x in aliases if x != "timestamp" and x in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    pc = df["close"].shift()
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    d = df["close"].diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["ret3"] = df["close"].pct_change(3)
    df["ret12"] = df["close"].pct_change(12)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"]
    df["hour_utc"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["spread_pips"] = (
        ((df["ask_close"] - df["bid_close"]) / 0.0001)
        if {"bid_close", "ask_close"}.issubset(df.columns)
        else np.nan
    )
    return df


def load_setups(path):
    raw = pd.read_csv(path)
    aliases = {
        "setup_id": ["setup_id", "id", "zone_id"],
        "timestamp": [
            "timestamp",
            "entry_timestamp",
            "entry_time",
            "setup_timestamp",
            "created_at",
        ],
        "direction": ["direction", "side", "trade_direction"],
        "entry": ["entry", "entry_price", "price"],
    }
    rename = {}
    for canonical, choices in aliases.items():
        col = find_col(raw, choices, canonical in {"timestamp", "direction"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    if "setup_id" not in df:
        df["setup_id"] = [f"setup_{i:06d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    d = df["direction"].astype(str).str.lower().str.strip()
    norm = pd.Series(pd.NA, index=df.index, dtype="object")
    norm.loc[d.isin(["long", "buy", "bull", "bullish", "demand", "1"])] = "long"
    norm.loc[d.isin(["short", "sell", "bear", "bearish", "supply", "-1"])] = "short"
    df["direction"] = norm
    if "entry" in df:
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
    return (
        df.dropna(subset=["timestamp", "direction"])
        .sort_values("timestamp")
        .drop_duplicates("setup_id")
        .reset_index(drop=True)
    )


def load_phase1(path):
    df = pd.read_csv(path)
    if "setup_id" not in df:
        raise ValueError("Phase 1 database must contain setup_id.")
    return df

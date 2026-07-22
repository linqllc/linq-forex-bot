from __future__ import annotations
from pathlib import Path
import pandas as pd


REQUIRED = {"open", "high", "low", "close"}


def load_candles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    time_col = next(
        (lower[x] for x in ("time", "timestamp", "datetime", "date") if x in lower),
        None,
    )
    if time_col is None:
        raise ValueError("CSV needs time/timestamp/datetime/date column.")

    rename = {time_col: "time"}
    for col in REQUIRED | {"volume"}:
        if col in lower:
            rename[lower[col]] = col
    df = df.rename(columns=rename)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
    for col in REQUIRED | ({"volume"} & set(df.columns)):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        raise ValueError(f"Invalid OHLC geometry in {int(bad.sum())} rows.")

    return df

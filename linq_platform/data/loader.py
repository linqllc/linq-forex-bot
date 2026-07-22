from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date"],
    "open": ["open", "mid_open"],
    "high": ["high", "mid_high"],
    "low": ["low", "mid_low"],
    "close": ["close", "mid_close"],
    "volume": ["volume", "tick_volume"],
}


def _locate_column(columns, candidates):
    lowered = {str(column).lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def load_oanda_m5(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Candle file not found: {path.resolve()}"
        )

    raw = pd.read_csv(path)

    rename_map = {}

    for standard_name, candidates in COLUMN_ALIASES.items():
        source = _locate_column(raw.columns, candidates)

        if source is None and standard_name != "volume":
            raise ValueError(
                f"Could not locate {standard_name!r}. "
                f"Available columns: {list(raw.columns)}"
            )

        if source is not None:
            rename_map[source] = standard_name

    candles = raw.rename(columns=rename_map).copy()

    if "volume" not in candles:
        candles["volume"] = 0.0

    candles["timestamp"] = pd.to_datetime(
        candles["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        candles[column] = pd.to_numeric(
            candles[column],
            errors="coerce",
        )

    candles = candles.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    candles = (
        candles.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    invalid = (
        (candles["high"] < candles["low"])
        | (candles["open"] > candles["high"])
        | (candles["open"] < candles["low"])
        | (candles["close"] > candles["high"])
        | (candles["close"] < candles["low"])
    )

    if invalid.any():
        raise ValueError(
            f"Detected {int(invalid.sum())} invalid OHLC rows."
        )

    candles = candles.set_index("timestamp")

    # Backtrader expects a timezone-naive DatetimeIndex.
    candles.index = candles.index.tz_convert("UTC").tz_localize(None)

    return candles[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)


def load_selected_signals(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {path.resolve()}"
        )

    signals = pd.read_csv(path)

    required = {
        "timestamp",
        "direction",
        "entry_price",
        "stop_price",
        "target_price",
    }

    missing = required.difference(signals.columns)

    if missing:
        raise ValueError(
            f"Signal file is missing columns: {sorted(missing)}"
        )

    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "entry_price",
        "stop_price",
        "target_price",
        "probability_1r",
        "gross_result_r",
        "net_result_r",
    ]:
        if column in signals:
            signals[column] = pd.to_numeric(
                signals[column],
                errors="coerce",
            )

    signals["direction"] = (
        signals["direction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    signals = signals[
        signals["direction"].isin(["long", "short"])
    ]

    signals = (
        signals.dropna(
            subset=[
                "timestamp",
                "entry_price",
                "stop_price",
                "target_price",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="first")
        .reset_index(drop=True)
    )

    return signals

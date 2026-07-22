from __future__ import annotations

from pathlib import Path
import pandas as pd


CANONICAL_ALIASES = {
    "time": (
        "time",
        "timestamp",
        "datetime",
        "date",
    ),
    "open": (
        "open",
        "o",
        "mid_o",
        "mid_open",
        "bid_o",
        "bid_open",
        "ask_o",
        "ask_open",
    ),
    "high": (
        "high",
        "h",
        "mid_h",
        "mid_high",
        "bid_h",
        "bid_high",
        "ask_h",
        "ask_high",
    ),
    "low": (
        "low",
        "l",
        "mid_l",
        "mid_low",
        "bid_l",
        "bid_low",
        "ask_l",
        "ask_low",
    ),
    "close": (
        "close",
        "c",
        "mid_c",
        "mid_close",
        "bid_c",
        "bid_close",
        "ask_c",
        "ask_close",
    ),
    "volume": (
        "volume",
        "vol",
        "v",
        "tick_volume",
    ),
}


def _normalized(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace(".", "_")
    )


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    available = {_normalized(column): column for column in df.columns}
    rename: dict[str, str] = {}

    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            candidate = _normalized(alias)
            if candidate in available:
                rename[available[candidate]] = canonical
                break

    return rename


def load_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Candle CSV not found: {path}")

    df = pd.read_csv(path)
    original_columns = list(df.columns)
    df = df.rename(columns=_resolve_columns(df))

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing OHLC columns: {sorted(missing)}. "
            f"Available columns: {original_columns}"
        )

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")

    numeric_columns = ["open", "high", "low", "close"]

    if "volume" in df.columns:
        numeric_columns.append("volume")

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    invalid = (
        df["high"] < df[["open", "close", "low"]].max(axis=1)
    ) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )

    if invalid.any():
        raise ValueError(
            f"Invalid OHLC geometry in {int(invalid.sum())} rows."
        )

    return df

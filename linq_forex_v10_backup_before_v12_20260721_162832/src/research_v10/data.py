from __future__ import annotations

from pathlib import Path
import pandas as pd


TIMESTAMP_CANDIDATES = (
    "timestamp", "time", "datetime", "date", "complete_time", "candle_time"
)

COLUMN_ALIASES = {
    "open": ("open", "o", "mid_open", "mid_o", "bid_open", "bid_o", "ask_open", "ask_o"),
    "high": ("high", "h", "mid_high", "mid_h", "bid_high", "bid_h", "ask_high", "ask_h"),
    "low": ("low", "l", "mid_low", "mid_l", "bid_low", "bid_l", "ask_low", "ask_l"),
    "close": ("close", "c", "mid_close", "mid_c", "bid_close", "bid_c", "ask_close", "ask_c"),
    "volume": ("volume", "v", "tick_volume"),
}


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_candles(path: str | Path, timestamp_column: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candle file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Candle CSV is empty.")

    columns = list(df.columns)
    ts_col = timestamp_column or _find_column(columns, TIMESTAMP_CANDIDATES)
    if ts_col is None:
        unnamed = [c for c in columns if c.lower().startswith("unnamed")]
        if unnamed:
            ts_col = unnamed[0]
        else:
            raise ValueError(
                "Could not identify timestamp column. Pass --timestamp-column explicitly."
            )

    rename_map: dict[str, str] = {ts_col: "timestamp"}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _find_column(columns, aliases)
        if found is None and canonical != "volume":
            raise ValueError(f"Missing required candle column: {canonical}")
        if found is not None:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)
    keep = ["timestamp", "open", "high", "low", "close"]
    if "volume" in df.columns:
        keep.append("volume")
    df = df[keep].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
          .drop_duplicates(subset=["timestamp"], keep="last")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} malformed OHLC rows.")

    return df

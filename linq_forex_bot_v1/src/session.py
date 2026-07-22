from __future__ import annotations
from datetime import time
from zoneinfo import ZoneInfo
import pandas as pd


def parse_clock(value: str) -> time:
    hour, minute = (int(piece) for piece in value.split(":"))
    return time(hour, minute)


def add_session_columns(df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    out = df.copy()
    timestamps = pd.to_datetime(out["time"], utc=True)
    local = timestamps.dt.tz_convert(ZoneInfo(timezone_name))
    out["local_time"] = local
    out["session_date"] = local.dt.date
    out["clock"] = local.dt.time
    return out


def calculate_opening_ranges(
    df: pd.DataFrame,
    start_clock: str,
    end_clock: str,
) -> pd.DataFrame:
    out = df.copy()
    start = parse_clock(start_clock)
    end = parse_clock(end_clock)
    mask = out["clock"].map(lambda t: start <= t < end)

    ranges = (
        out.loc[mask]
        .groupby("session_date")
        .agg(
            opening_range_high=("mid_high", "max"),
            opening_range_low=("mid_low", "min"),
            opening_range_candles=("time", "count"),
        )
        .reset_index()
    )
    out = out.merge(ranges, on="session_date", how="left")
    return out

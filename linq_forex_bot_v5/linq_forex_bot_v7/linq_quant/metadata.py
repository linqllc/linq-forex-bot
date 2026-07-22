from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def infer_data_source(path: Path) -> str:
    name = path.name.lower()

    if "synthetic" in name or "demo" in name:
        return "synthetic_demo"

    if "oanda" in name:
        return "oanda"

    return "unknown"


def infer_instrument(path: Path) -> str:
    match = re.match(r"([A-Z]{3}_[A-Z]{3})", path.name.upper())
    return match.group(1) if match else "UNKNOWN"


def infer_strategy_version(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]

    for part in reversed(lowered_parts):
        match = re.search(r"v(\d+(?:\.\d+)?)", part)
        if match:
            return f"v{match.group(1)}"

    return "unknown"


def infer_timeframe(frame: pd.DataFrame) -> str:
    if "timeframe" in frame.columns:
        values = frame["timeframe"].dropna().astype(str)

        if not values.empty:
            return values.mode().iloc[0]

    timestamp_column = None

    for candidate in ("timestamp", "time", "entry_time"):
        if candidate in frame.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        return "unknown"

    timestamps = pd.to_datetime(
        frame[timestamp_column],
        utc=True,
        errors="coerce",
    ).dropna().sort_values()

    if len(timestamps) < 2:
        return "unknown"

    median_minutes = (
        timestamps.diff()
        .dropna()
        .dt.total_seconds()
        .median()
        / 60
    )

    if median_minutes <= 6:
        return "M5"

    if median_minutes <= 16:
        return "M15"

    if median_minutes <= 31:
        return "M30"

    if median_minutes <= 61:
        return "H1"

    if median_minutes <= 241:
        return "H4"

    return "custom"


def classify_session(timestamp: object) -> str:
    parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")

    if pd.isna(parsed):
        return "unknown"

    hour = int(parsed.hour)

    # Simplified UTC windows. These are research labels, not broker-session
    # guarantees, and daylight-saving differences should be studied separately.
    london = 7 <= hour < 16
    new_york = 12 <= hour < 21
    asia = hour >= 23 or hour < 8

    if london and new_york:
        return "london_new_york_overlap"

    if london:
        return "london"

    if new_york:
        return "new_york"

    if asia:
        return "asia"

    return "off_session"


def create_run_id(
    path: Path,
    data_source: str,
    strategy_version: str,
    timeframe: str,
) -> str:
    stat = path.stat()

    identity = "|".join(
        [
            str(path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            data_source,
            strategy_version,
            timeframe,
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:20]

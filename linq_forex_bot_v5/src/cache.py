from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def cache_path(root: str | Path, instrument: str, granularity: str) -> Path:
    return Path(root) / f"{instrument}_{granularity}.csv"


def load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["time"])
    if frame.empty:
        return None
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def merge_cache(existing: pd.DataFrame | None, fresh: pd.DataFrame, path: Path) -> pd.DataFrame:
    parts = [x for x in (existing, fresh) if x is not None and not x.empty]
    combined = pd.concat(parts, ignore_index=True).sort_values("time").drop_duplicates("time")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined.reset_index(drop=True)

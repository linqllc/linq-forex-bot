from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
import os
import time

import pandas as pd
import requests


class OandaError(RuntimeError):
    pass


class OandaClient:
    """Minimal read-only OANDA v20 client for historical candle research."""

    def __init__(self, token: str | None = None, environment: str | None = None):
        self.token = token or os.getenv("OANDA_TOKEN")
        self.environment = (environment or os.getenv("OANDA_ENV", "practice")).lower()

        if not self.token:
            raise OandaError(
                "Missing OANDA_TOKEN. Copy .env.example to .env and add a practice token."
            )
        if self.environment not in {"practice", "live"}:
            raise OandaError("OANDA_ENV must be 'practice' or 'live'.")

        host = (
            "https://api-fxpractice.oanda.com"
            if self.environment == "practice"
            else "https://api-fxtrade.oanda.com"
        )
        self.base_url = f"{host}/v3"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            }
        )

    def get_candles(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        granularity: str = "M5",
        price: str = "MBA",
        max_per_request: int = 5000,
    ) -> pd.DataFrame:
        """Download complete candles in chunks and return bid/mid/ask OHLC data."""
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes.")
        if start >= end:
            raise ValueError("start must be earlier than end.")

        rows: list[dict] = []
        cursor = start.astimezone(timezone.utc)

        while cursor < end.astimezone(timezone.utc):
            params = {
                "from": cursor.isoformat().replace("+00:00", "Z"),
                "to": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "granularity": granularity,
                "price": price,
                "count": max_per_request,
            }
            # OANDA does not permit count with both from/to on every endpoint version.
            # Use from + count, then advance from the last returned candle.
            params.pop("to")
            response = self.session.get(
                f"{self.base_url}/instruments/{instrument}/candles",
                params=params,
                timeout=30,
            )
            if response.status_code != 200:
                raise OandaError(
                    f"OANDA request failed ({response.status_code}): {response.text[:500]}"
                )

            payload = response.json()
            candles = payload.get("candles", [])
            if not candles:
                break

            last_time = None
            for candle in candles:
                ts = pd.Timestamp(candle["time"])
                last_time = ts
                if ts >= pd.Timestamp(end):
                    break
                if not candle.get("complete", False):
                    continue
                row = {
                    "time": ts,
                    "volume": int(candle.get("volume", 0)),
                    "complete": bool(candle.get("complete", False)),
                }
                for prefix, key in (("mid", "mid"), ("bid", "bid"), ("ask", "ask")):
                    values = candle.get(key)
                    if values:
                        row.update(
                            {
                                f"{prefix}_open": float(values["o"]),
                                f"{prefix}_high": float(values["h"]),
                                f"{prefix}_low": float(values["l"]),
                                f"{prefix}_close": float(values["c"]),
                            }
                        )
                rows.append(row)

            if last_time is None:
                break
            next_cursor = last_time.to_pydatetime() + pd.Timedelta(seconds=1)
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.05)

        if not rows:
            raise OandaError(f"No candles returned for {instrument}.")

        frame = pd.DataFrame(rows).drop_duplicates(subset=["time"]).sort_values("time")
        frame = frame[frame["time"] < pd.Timestamp(end)]
        return frame.reset_index(drop=True)

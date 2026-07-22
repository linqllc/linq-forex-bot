from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests


PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download paginated OANDA M1 candles."
    )

    parser.add_argument(
        "--instrument",
        default="EUR_USD",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="UTC ISO timestamp",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="UTC ISO timestamp",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--environment",
        choices=["practice", "live"],
        default="practice",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=72,
        help="Hours requested per API call.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
    )

    return parser.parse_args()


def utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def format_oanda_time(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def request_chunk(
    session: requests.Session,
    base_url: str,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    url = (
        f"{base_url}/v3/instruments/"
        f"{instrument}/candles"
    )

    params = {
        "from": format_oanda_time(start),
        "to": format_oanda_time(end),
        "granularity": "M1",
        "price": "MBA",
        "smooth": "false",
        "includeFirst": "true",
    }

    response = session.get(
        url,
        params=params,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OANDA request failed "
            f"{response.status_code}: "
            f"{response.text}"
        )

    payload = response.json()
    return payload.get("candles", [])


def candle_to_row(candle: dict) -> dict:
    mid = candle.get("mid", {})
    bid = candle.get("bid", {})
    ask = candle.get("ask", {})

    return {
        "time": candle["time"],
        "volume": candle.get("volume"),
        "complete": candle.get("complete"),
        "mid_open": mid.get("o"),
        "mid_high": mid.get("h"),
        "mid_low": mid.get("l"),
        "mid_close": mid.get("c"),
        "bid_open": bid.get("o"),
        "bid_high": bid.get("h"),
        "bid_low": bid.get("l"),
        "bid_close": bid.get("c"),
        "ask_open": ask.get("o"),
        "ask_high": ask.get("h"),
        "ask_low": ask.get("l"),
        "ask_close": ask.get("c"),
    }


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No candles downloaded.")

    required = {
        "time",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    timestamps = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise RuntimeError(
            "One or more timestamps could not be parsed."
        )

    if timestamps.duplicated().any():
        raise RuntimeError(
            "Duplicate timestamps remain after processing."
        )

    if not timestamps.is_monotonic_increasing:
        raise RuntimeError(
            "Timestamps are not sorted."
        )


def main() -> None:
    args = parse_args()

    token = os.environ.get("OANDA_API_TOKEN")

    if not token:
        raise RuntimeError(
            "OANDA_API_TOKEN is not set.\n"
            "Run:\n"
            "export OANDA_API_TOKEN='your-token'"
        )

    start = utc_timestamp(args.start)
    end = utc_timestamp(args.end)

    if end <= start:
        raise ValueError(
            "--end must be later than --start"
        )

    base_url = (
        PRACTICE_URL
        if args.environment == "practice"
        else LIVE_URL
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept-Datetime-Format": "RFC3339",
        }
    )

    chunk_size = pd.Timedelta(
        hours=args.chunk_hours
    )

    current = start
    rows: list[dict] = []
    request_number = 0

    while current < end:
        chunk_end = min(
            current + chunk_size,
            end,
        )

        request_number += 1

        print(
            f"[{request_number}] "
            f"{current.isoformat()} → "
            f"{chunk_end.isoformat()}"
        )

        candles = request_chunk(
            session=session,
            base_url=base_url,
            instrument=args.instrument,
            start=current,
            end=chunk_end,
        )

        complete_candles = [
            candle
            for candle in candles
            if candle.get("complete", False)
        ]

        rows.extend(
            candle_to_row(candle)
            for candle in complete_candles
        )

        print(
            f"    received={len(candles)} "
            f"complete={len(complete_candles)} "
            f"accumulated={len(rows)}"
        )

        current = chunk_end
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No candle rows were returned."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = (
        df.sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    numeric_columns = [
        column
        for column in df.columns
        if column not in {
            "time",
            "complete",
        }
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    validate(df)

    df.to_csv(
        output,
        index=False,
    )

    gaps = (
        df["time"]
        .diff()
        .value_counts()
        .head(10)
    )

    print("\nDownload complete")
    print(f"Output: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['time'].min()}")
    print(f"End: {df['time'].max()}")
    print("\nMost common gaps:")
    print(gaps)

    dominant_gap = gaps.index[0]

    if dominant_gap != pd.Timedelta(minutes=1):
        raise RuntimeError(
            "Validation failed: dominant candle "
            f"gap is {dominant_gap}, not one minute."
        )

    print("\n✅ Genuine M1 interval verified")


if __name__ == "__main__":
    main()

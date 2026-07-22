from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.demo_data import generate_demo_data
from src.indicators import add_indicators
from src.metrics import performance_summary
from src.oanda_client import OandaClient
from src.session import add_session_columns, calculate_opening_ranges
from src.strategy import run_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="LINQ Forex Supply/Demand research bot")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic candles to verify installation; not for strategy evaluation.",
    )
    parser.add_argument("--config", default="config/strategy.yaml")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    instrument = args.instrument.upper()

    if args.demo:
        candles = generate_demo_data(days=args.days)
        data_source = "synthetic_demo"
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        client = OandaClient()
        candles = client.get_candles(
            instrument=instrument,
            start=start,
            end=end,
            granularity=config["granularity"],
        )
        data_source = "oanda"

    candles = add_session_columns(candles, config["timezone"])
    candles = add_indicators(
        candles,
        atr_period=config["impulse"]["atr_period"],
        body_lookback=config["impulse"]["lookback_bodies"],
    )
    candles = calculate_opening_ranges(
        candles,
        config["session"]["opening_range_start"],
        config["session"]["opening_range_end"],
    )

    trades = run_strategy(candles, instrument, config)
    summary = performance_summary(trades)
    summary["instrument"] = instrument
    summary["data_source"] = data_source
    summary["days_requested"] = args.days

    reports = Path("reports")
    processed = Path("data/processed")
    reports.mkdir(exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    candle_path = processed / f"{instrument}_{data_source}_candles.csv"
    trade_path = reports / f"{instrument}_{data_source}_trades.csv"
    summary_path = reports / f"{instrument}_{data_source}_summary.json"

    candles.to_csv(candle_path, index=False)
    trades.to_csv(trade_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved candles: {candle_path}")
    print(f"Saved trades:  {trade_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

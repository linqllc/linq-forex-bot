from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.cache import cache_path, load_cached, merge_cache
from src.config import load_config
from src.demo_data import generate_demo_data
from src.indicators import add_indicators
from src.market_structure import add_market_structure
from src.oanda_client import OandaClient
from src.optimizer import optimize_setups, reward_grid
from src.progress import ProgressBar, StageTimer
from src.session import add_session_columns, calculate_opening_ranges
from src.setups import scan_setups
from src.walk_forward import walk_forward_report


def prepare(candles, c):
    stages = [
        ("Session columns", lambda x: add_session_columns(x, c["timezone"])),
        ("Indicators", lambda x: add_indicators(x, c["impulse"]["atr_period"], c["impulse"]["lookback_bodies"])),
        ("Market structure", lambda x: add_market_structure(x, c["structure"]["fast_ema"], c["structure"]["slow_ema"], c["structure"]["swing_window"])),
        ("Opening ranges", lambda x: calculate_opening_ranges(x, c["session"]["opening_range_start"], c["session"]["opening_range_end"])),
    ]
    x = candles
    bar = ProgressBar("Preparing candles", len(stages))
    for idx, (name, fn) in enumerate(stages, start=1):
        x = fn(x)
        bar.update(idx, detail=name, force=True)
    return x


def get_data(instrument, days, config, demo, refresh):
    if demo:
        timer = StageTimer(f"Generating {instrument} demo data")
        frame = generate_demo_data(days)
        timer.done(f"{len(frame):,} candles")
        return frame, "synthetic_demo"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    path = cache_path("data/cache", instrument, config["granularity"])
    cached = load_cached(path)
    if cached is not None and not refresh and cached.time.min() <= pd.Timestamp(start) and cached.time.max() >= pd.Timestamp(end-timedelta(minutes=10)):
        frame = cached[(cached.time >= pd.Timestamp(start)) & (cached.time < pd.Timestamp(end))].reset_index(drop=True)
        print(f"✓ Cache hit: loaded {len(frame):,} {instrument} candles without downloading")
        return frame, "oanda_cache"

    progress = ProgressBar(f"Downloading {instrument}", max(1, int((end-start).total_seconds())))
    client = OandaClient()
    fresh = client.get_candles(
        instrument,
        start,
        end,
        config["granularity"],
        progress_callback=lambda current, total, detail: progress.update(current, detail=detail),
    )
    progress.close(f"{len(fresh):,} candles")
    timer = StageTimer(f"Updating {instrument} cache")
    combined = merge_cache(cached, fresh, path)
    timer.done(f"{len(combined):,} cached candles")
    return combined[(combined.time >= pd.Timestamp(start)) & (combined.time < pd.Timestamp(end))].reset_index(drop=True), "oanda"


def main():
    p = argparse.ArgumentParser(description="LINQ Forex Bot v3.1: scored setups, fast R optimization, progress and ETA")
    p.add_argument("--instruments", nargs="+", default=None)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--min-r", type=float, default=None)
    p.add_argument("--max-r", type=float, default=None)
    p.add_argument("--step", type=float, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--config", default="config/strategy.yaml")
    args = p.parse_args()
    load_dotenv()
    c = load_config(args.config)
    oc = c["optimizer"]
    ratios = reward_grid(args.min_r or oc["minimum_r"], args.max_r or oc["maximum_r"], args.step or oc["step_r"])
    instruments = [x.upper() for x in (args.instruments or c["instruments"])]
    out = Path("reports/v3")
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    overall = ProgressBar("Overall pair progress", len(instruments))

    for pair_number, instrument in enumerate(instruments, start=1):
        print(f"\n{'='*72}\nLINQ Forex Bot v3.1 | {instrument} | pair {pair_number}/{len(instruments)}\n{'='*72}")
        raw, source = get_data(instrument, args.days, c, args.demo, args.refresh)
        candles = prepare(raw, c)

        scan_bar = ProgressBar(f"Scanning {instrument}", max(1, len(candles)-1))
        setups = scan_setups(
            candles,
            instrument,
            c,
            progress_callback=lambda current, total, detail: scan_bar.update(current, detail=detail),
        )
        scan_bar.close(f"{len(setups)} qualified setups")

        ratio_bar = ProgressBar(f"Optimizing {instrument}", len(ratios))
        ratio_results, selection = optimize_setups(
            setups,
            ratios,
            oc["minimum_trades"],
            progress_callback=lambda current, total, detail: ratio_bar.update(current, detail=detail, force=True),
        )
        ratio_bar.close(f"best {selection['best_overall_ratio']:.2f}R")

        wf_timer = StageTimer(f"Walk-forward validation for {instrument}")
        wf, wf_summary = walk_forward_report(setups, ratios, c)
        wf_timer.done()

        summary = {
            "instrument": instrument,
            "data_source": source,
            "days_requested": args.days,
            "qualified_setups": len(setups),
            **selection,
            **wf_summary,
        }
        save_timer = StageTimer(f"Saving {instrument} reports")
        setups.to_csv(out/f"{instrument}_{source}_setups.csv", index=False)
        ratio_results.to_csv(out/f"{instrument}_{source}_ratio_results.csv", index=False)
        wf.to_csv(out/f"{instrument}_{source}_walk_forward.csv", index=False)
        (out/f"{instrument}_{source}_summary.json").write_text(json.dumps(summary, indent=2))
        save_timer.done(str(out))
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
        overall.update(pair_number, detail=f"finished {instrument}", force=True)

    pd.DataFrame(summaries).to_csv(out/"pair_summary.csv", index=False)
    overall.close(f"reports saved to {out}")
    print(f"\n✓ All work complete. Reports: {out}")


if __name__ == "__main__":
    main()

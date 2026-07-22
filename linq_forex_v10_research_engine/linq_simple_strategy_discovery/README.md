# LINQ Simple Strategy Discovery Engine 1.1

This standalone historical strategy-search backtester searches for simple trading rules, ranks them using discovery and validation data, and opens the locked test only after finalists have been frozen.

## What changed from 1.0

- The locked test is no longer used to rank or select strategies.
- OANDA midpoint columns such as `mid_open`, `mid_high`, `mid_low`, and `mid_close` are supported.
- Spread and slippage are applied on both entry and exit for stop, target, and time exits.
- Finalists remain in their original discovery ranking after locked-test evaluation.
- Locked-test qualification requires positive expectancy, enough trades, and a minimum profit factor.

## Search space

The engine tests 5,184 combinations across:

- EMA crossover
- MACD crossover
- RSI reversal
- Bollinger Band re-entry
- Donchian breakout
- Supertrend flip
- Long, short, or both directions
- No trend filter, 200 EMA filter, or 50/200 EMA alignment
- All hours, Asia, London open, or New York
- Optional above-median volatility filter
- 1.0, 1.5, or 2.0 ATR stop
- 1.0R, 1.5R, 2.0R, or 3.0R target

Signals are evaluated on completed candles and entries occur at the next candle open. Overlapping positions are blocked.

## Install

```bash
cd ~/Desktop/linq_forex_bot/linq_forex_v10_research_engine
unzip ~/Downloads/linq_simple_strategy_discovery_v1_1.zip
pip install -r linq_simple_strategy_discovery/requirements.txt
```

## Run

```bash
python3 linq_simple_strategy_discovery/run_simple_strategy_discovery.py \
  --candles "/Users/rgoode/Desktop/linq_forex_bot/linq_forex_bot_v5/linq_forex_bot_v7/data/cache/EUR_USD_M5.csv" \
  --instrument EUR_USD \
  --report-dir reports/simple_discovery_v1_1 \
  --pip-size 0.0001 \
  --spread-pips 0.8 \
  --slippage-pips 0.2 \
  --top-n 25
```

## Outputs

- `EUR_USD_discovery_ranking.csv` — all strategies ranked using training and validation only
- `EUR_USD_finalist_locked_test.csv` — frozen finalists evaluated on the untouched test period
- `EUR_USD_summary.json` — assumptions, gates, and plain-English finalist summaries
- `finalist_XX_trades.csv` — complete trade logs for each finalist

## Important

A locked-test pass is not permission to deploy live. The next phase should add parameter-neighbor stress tests, walk-forward evaluation, higher-cost stress tests, Monte Carlo sequence analysis, and demo execution validation.

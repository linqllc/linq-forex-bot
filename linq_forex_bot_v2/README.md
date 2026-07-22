# LINQ Forex Supply & Demand Research Bot v1

This is a **research-only** first build. It downloads five-minute forex candles,
calculates the 9:30-9:45 a.m. New York opening range, detects objective impulse
breakouts, creates a zone from the final opposing candle, waits for the first
confirmed retest, and simulates a fixed-R trade.

It does **not** place live or practice orders.

## Current baseline rules

- Instruments: EUR/USD, GBP/USD, USD/JPY, AUD/USD
- Timeframe: M5
- Opening range: 9:30-9:45 a.m. America/New_York
- Entries: 9:45 a.m.-12:00 p.m.
- Impulse body: at least 1.25x prior 20-candle median body
- Body fraction: at least 60% of candle range
- Displacement: at least 1 ATR
- Zone: final opposite-colored candle before impulse
- Entry: first retest plus directional rejection candle
- Stop: beyond zone plus 0.10 ATR
- Target: 2R
- Maximum: one trade per pair per day
- Execution model includes configurable spread and slippage

## Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verify installation without credentials

This runs synthetic data through the complete pipeline. Synthetic results are
only a software test and must never be treated as evidence of profitability.

```bash
python run_research.py --demo --instrument EUR_USD --days 90
```

## Connect an OANDA practice account

1. Open an OANDA v20 practice account.
2. Generate a personal access token.
3. Copy `.env.example` to `.env`.
4. Add your token:

```text
OANDA_TOKEN=your_private_token
OANDA_ENV=practice
```

Never send the token in chat, commit it to GitHub, or share it with anyone.

Run:

```bash
python run_research.py --instrument EUR_USD --days 180
```

Outputs are saved in:

- `data/processed/`: processed candle data
- `reports/`: simulated trades and performance summary

## Important limitations in v1

- The opening range is copied from the source strategy and has not yet been
  proven to be the best forex session definition.
- OANDA candle requests may require chunk tuning for very long histories.
- The backtester uses a conservative same-candle rule: if stop and target are
  both touched, the stop is counted first.
- News filtering, correlated-exposure limits, chart exports, walk-forward
  testing, London-session testing, and practice-order execution are later phases.
- No result is guaranteed. Do not fund a live account based on an in-sample test.

## Version 2 additions
- Completed-candle EMA trend, confirmed swing highs/lows, and break-of-structure columns.
- Pair-by-pair reward/risk optimizer from 1R through 5R in 0.25R increments.
- Separate selections for best overall, net profit, expectancy, and win rate.
- CSV and JSON comparison reports in `reports/optimizer/`.

Run the installation test:
```bash
python run_optimizer.py --demo --instruments EUR_USD --days 90
```

Run real OANDA research:
```bash
python run_optimizer.py --instruments EUR_USD GBP_USD USD_JPY AUD_USD --days 365
```

Optimizer results are research output, not proof of future profitability. Validate on untouched periods and paper trading before live use.

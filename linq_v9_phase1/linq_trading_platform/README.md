# LINQ Trading Platform V10

Backtrader-based migration of the LINQ Market Intelligence Engine.

## Current milestone

V10 Foundation:

- Loads OANDA EUR/USD M5 data
- Normalizes OANDA mid-price columns
- Loads Phase 4.1 model-selected trades
- Executes Backtrader bracket orders
- Uses the original stop and target prices
- Tracks fills, closed trades, equity and drawdown
- Saves CSV and JSON reports

This first build is a migration/parity test. It does not retrain the AI model
inside Backtrader yet.

## Run

From the linq_v9_phase1 directory:

    source .venv/bin/activate
    cd linq_trading_platform
    python main.py

## Output

    reports/v10/

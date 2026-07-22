# LINQ V9 Phase 1

Copy `run_v9_phase1.py` into the root of `linq_forex_bot_v7`.

Run:

```bash
python -m pip install pandas numpy
python run_v9_phase1.py
```

It reads:

- `data/cache/EUR_USD_M5.csv`
- `reports/v7/EUR_USD_automatic_setups.csv`

It creates:

- `reports/v9/EUR_USD_market_database.csv`
- `reports/v9/EUR_USD_market_database_summary.json`
- `reports/v9/EUR_USD_market_database_report.html`

This phase builds leakage-safe historical examples and labels each setup at 1R, 1.25R, 1.5R, 2R, and 3R. The next phase trains the adaptive target-selection model from this database.

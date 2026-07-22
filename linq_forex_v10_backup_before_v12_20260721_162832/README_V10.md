# LINQ Forex Research Engine — V10

This is a separate, single-strategy research module for the existing LINQ Forex project.

It does **not** replace V7 or V9. It reuses the existing candle cache and adds:

- objective displacement/pullback setup detection
- ATR, EMA, volatility, session, FVG, trend, and structure-distance features
- MFE/MAE and fixed-R outcome labeling
- chronological train/test split
- single-feature ranking
- multi-rule out-of-sample validation
- CSV and JSON research reports

## Install into the existing project

Copy these files into the root of:

`/Users/rgoode/Desktop/linq_forex_bot/linq_forex_bot_v5/linq_forex_bot_v7`

The final tree should include:

```text
run_v10_research.py
README_V10.md
src/research_v10/
tests/test_v10_research.py
```

## Run tests

```bash
python -m pytest tests/test_v10_research.py -q
```

## Run EUR/USD research

```bash
python run_v10_research.py   --instrument EUR_USD   --csv data/cache/EUR_USD_M5.csv   --report-dir reports/v10
```

## Reports

```text
reports/v10/EUR_USD_v10_setups.csv
reports/v10/EUR_USD_v10_feature_ranking.csv
reports/v10/EUR_USD_v10_rule_validation.csv
reports/v10/EUR_USD_v10_summary.json
```

## Important research controls

- The target is clamped to a minimum of 1R.
- Train/test splitting is chronological, not random.
- Same-bar stop-and-target collisions default to `stop_first`.
- The winning rule is not ready for live trading until it survives another untouched dataset and paper trading.

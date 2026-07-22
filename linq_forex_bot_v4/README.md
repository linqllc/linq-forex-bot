# LINQ Forex Research Bot v4

A research-first forex backtesting platform built on the working v3 pipeline.

## What v4 adds

- ML-ready feature dataset for every qualified setup
- Outcome labels for every target ratio in the configured search grid
- Per-ratio hit probability and theoretical expectancy tables
- Performance breakdowns by direction, weekday, hour, trend alignment, BOS, and score band
- Feature summary statistics
- Existing OANDA cache, progress bars, ratio optimization, and walk-forward reports

## Run

```bash
python run_v4.py --demo --instruments EUR_USD --days 90
python run_v4.py --instruments EUR_USD --days 365 --min-r 1 --max-r 5 --step 0.25
```

## Main v4 reports

Saved under `reports/v4/`:

- `*_ml_dataset.csv`
- `*_target_probabilities.csv`
- `*_condition_breakdown.csv`
- `*_feature_summary.csv`
- Existing setup, optimizer, walk-forward, and summary reports

## Important

This is research software, not proof of profitability. The ML dataset is intentionally generated without using future information in the feature columns; future outcomes are stored only as labels.

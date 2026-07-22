# LINQ Forex Intelligence Bot v5

A research-only forex platform with real OANDA candles, cached data, transparent market engines, reward/risk optimization, walk-forward reports, and an experimental chronological probability model.

## What v5 adds

- Trend engine score
- Market-structure engine score
- Supply/demand engine score
- Liquidity-sweep engine score
- Volatility-regime engine score
- Session-quality engine score
- Weighted confluence score
- Experimental logistic probability models for each R target
- Chronological train/test split
- Brier score and AUC reports
- Dynamic target recommendations based on estimated expectancy
- Model coefficients for transparency

The model refuses to claim training success when there are too few rows or only one outcome class. These outputs are research estimates, not evidence that a strategy is profitable.

## Setup

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Demo

```bash
python run_v5.py --demo --instruments EUR_USD --days 90 --min-r 2 --max-r 4 --step 0.5
```

## Real research

```bash
python run_v5.py --instruments EUR_USD --days 365 --min-r 1 --max-r 5 --step 0.25
```

Reports are saved in `reports/v5/`.

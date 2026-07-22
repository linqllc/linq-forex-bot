# LINQ Forex Bot v3

Research-only forex strategy platform. This version scans each market once, scores supply/demand setups, stores maximum favorable excursion, optimizes 1R–5R targets rapidly, caches OANDA candles, and validates the selected ratio on chronological train/validation/test splits.

## Setup
```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Add your OANDA practice token to `.env`.

## Quick demo
```bash
python run_v3.py --demo --instruments EUR_USD --days 90 --min-r 1 --max-r 5 --step 0.25
```

## Real data
```bash
python run_v3.py --instruments EUR_USD --days 365
```
Second runs reuse `data/cache`, so they are much faster. Use `--refresh` to redownload.

## Important
This is research software, not proof of profitability. Do not use it for live trading until results survive out-of-sample and paper-trading validation.

## Progress bars and estimated completion time (v3.1)

Long-running operations now display:

- Live percentage complete
- Elapsed time
- Estimated time remaining (ETA)
- Candle and setup counts
- Current reward-to-risk ratio under test
- Overall progress across multiple currency pairs

Example:

```text
Scanning EUR_USD          [██████████████░░░░░░░░░░░░░░] 51.20% | elapsed 0:00:14 | ETA 0:00:13 | 18 setups found
Optimizing EUR_USD        [████████████████████░░░░░░░░] 75.00% | elapsed 0:00:01 | ETA 0:00:00 | testing 4.00R
```

The ETA becomes more accurate after the first several percent of each stage. Cached OANDA data skips the download stage and displays a cache-hit message.

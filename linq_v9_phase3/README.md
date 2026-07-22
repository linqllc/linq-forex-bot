# LINQ Market Intelligence Engine — V9 Phase 3

Phase 3 performs a locked walk-forward validation of the current baseline and compares it with a structure-aware stop.

## Locked baseline

- Minimum model probability: 65%
- Stop: 2.0 ATR
- Target: 1.0R
- Initial training sample: 40 setups
- Expanding walk-forward predictions
- No trailing stop
- No partial exit

## Structure-aware comparison

For each setup, the engine calculates:

1. A 2 ATR volatility stop.
2. A structure stop based on the prior 12 completed M5 candles.
3. The zone or existing invalidation level when available.
4. A 0.10 ATR buffer beyond structure.

The structure-aware strategy uses the farther valid stop so that the stop is not placed inside obvious recent structure.

## Install

```bash
python -m pip install scikit-learn
```

## Run

Place `run_v9_phase3.py` in the same `linq_v9_phase1` directory that contains `data/` and `reports/`.

```bash
python -m py_compile run_v9_phase3.py && python run_v9_phase3.py
```

## Report

```bash
open reports/v9_phase3/EUR_USD_phase3_report.html
```

## Important

This remains research with a small historical sample. The structure-aware strategy is a comparison against the locked baseline; it does not retroactively replace the Phase 2 winner.

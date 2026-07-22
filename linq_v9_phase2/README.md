# LINQ V9 Phase 2

This phase evaluates alternative stop distances and tests whether a walk-forward model can select a higher-win-rate subset of setups at a minimum 1:1 reward-to-risk ratio.

## Stop methods tested

- Existing V7 stop
- 0.75 ATR
- 1.00 ATR
- 1.25 ATR
- 1.50 ATR
- 2.00 ATR

## Run

Place `run_v9_phase2.py` in the same high-level `linq_v9_phase1` folder containing `data` and `reports`.

```bash
python -m pip install scikit-learn
python run_v9_phase2.py
```

## Output

The report is saved to:

```text
reports/v9_phase2/EUR_USD_phase2_report.html
```

Open it with:

```bash
open reports/v9_phase2/EUR_USD_phase2_report.html
```

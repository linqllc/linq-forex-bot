# LINQ V9 Phase 4

This refactors the project into a small package and runs a frozen final-holdout validation.

Frozen rules:
- 65% probability threshold
- 2 ATR stop
- 1R target
- final 20 usable setups reserved as holdout
- spread and 0.10 pip slippage per side deducted
- no strategy comparison or retuning

Copy `run_v9_phase4.py` and the `linq_engine/` folder into your existing `linq_v9_phase1` directory.

Run:

```bash
python -m py_compile run_v9_phase4.py
python run_v9_phase4.py
```

Open:

```bash
open reports/v9_phase4/EUR_USD_phase4_report.html
```

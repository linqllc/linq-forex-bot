# LINQ Forex Bot

Research-first algorithmic trading platform for reproducible signal generation, Backtrader execution, parity validation, walk-forward research, paper trading, and later live execution.

## Current status

- Frozen Phase 4.1 reference engine preserved under `legacy/phase4_reference/`
- Native intelligence modules under `linq/intelligence/`
- Versioned run/report directories
- GitHub Actions test workflow
- Safe master-folder installer that preserves existing files

## Quick start

```bash
cd ~/Desktop/linq_forex_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run native research after your existing Backtrader files have been migrated:

```bash
python scripts/run_research.py --label baseline
```

## Safety

This repository is research software. A profitable backtest does not guarantee live profitability. Live execution should remain disabled until paper-trading, risk limits, reconciliation, and kill-switch tests pass.

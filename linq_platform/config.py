from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "EUR_USD"
    starting_cash: float = 100_000.0

    # One standard lot is intentionally not used during migration.
    # This fixed unit size makes parity/debugging easier.
    fixed_units: int = 10_000

    pip_size: float = 0.0001
    slippage_pips_each_side: float = 0.10

    # Backtrader data/execution behavior
    preload: bool = True
    runonce: bool = True

    # Keep one open position maximum during the initial migration.
    maximum_open_positions: int = 1

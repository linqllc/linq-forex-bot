from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Phase4Config:
    pair: str = "EUR_USD"
    probability_threshold: float = 0.65
    stop_atr: float = 2.0
    minimum_stop_pips: float = 8.0
    target_r: float = 1.0
    forward_bars: int = 288
    holdout_trades: int = 20
    minimum_training_rows: int = 40
    slippage_pips_each_side: float = 0.10
    maximum_spread_pips: float | None = 2.5
    maximum_spread_to_stop_ratio: float = 0.20
    pip_size: float = 0.0001

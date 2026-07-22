from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    instrument: str = "EUR_USD"
    timestamp_column: str | None = None
    pip_size: float = 0.0001

    # Candidate setup: displacement followed by pullback.
    atr_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 200
    displacement_atr_min: float = 1.25
    displacement_body_ratio_min: float = 0.65
    pullback_min: float = 0.25
    pullback_max: float = 0.75
    max_pullback_bars: int = 24

    # Outcome labeling.
    stop_atr: float = 1.25
    target_r: float = 1.5
    max_holding_bars: int = 96
    intrabar_policy: str = "stop_first"

    # Research split/search.
    train_fraction: float = 0.70
    min_train_trades: int = 20
    min_test_trades: int = 8
    top_rules: int = 20

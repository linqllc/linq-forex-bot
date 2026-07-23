"""Native signal-generation strategies."""

from .indicator_baseline import (
    EmaAtrStrategyConfig,
    calculate_atr,
    calculate_ema,
    calculate_true_ranges,
    generate_ema_atr_signals,
)

__all__ = [
    "EmaAtrStrategyConfig",
    "calculate_atr",
    "calculate_ema",
    "calculate_true_ranges",
    "generate_ema_atr_signals",
]

"""Native backtesting tools for the LINQ platform."""

from .config import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
)
from .execution import (
    calculate_entry_fill,
    calculate_exit_fill,
    open_position,
)
from .models import (
    Candle,
    ExitReason,
    OrderType,
    Position,
    PositionStatus,
    Trade,
    TradeDirection,
    TradeSignal,
)
from .risk import (
    calculate_position_size,
    calculate_target_price,
)
from .simulator import (
    calculate_gross_r,
    close_at_end_of_data,
    close_position,
    evaluate_position_on_candle,
    simulate_position,
)

__all__ = [
    "BacktestConfig",
    "Candle",
    "ExecutionConfig",
    "ExitReason",
    "OrderType",
    "Position",
    "PositionStatus",
    "RiskConfig",
    "StrategyConfig",
    "Trade",
    "TradeDirection",
    "TradeSignal",
    "calculate_entry_fill",
    "calculate_exit_fill",
    "calculate_gross_r",
    "calculate_position_size",
    "calculate_target_price",
    "close_at_end_of_data",
    "close_position",
    "evaluate_position_on_candle",
    "open_position",
    "simulate_position",
]

"""Native backtesting tools for the LINQ platform."""

from .config import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
)
from .data_loader import (
    load_candles_csv,
    load_signals_csv,
)
from .engine import BacktestResult, run_backtest
from .execution import (
    calculate_entry_fill,
    calculate_exit_fill,
    open_position,
)
from .metrics import (
    BacktestMetrics,
    build_equity_curve_r,
    calculate_backtest_metrics,
    calculate_maximum_drawdown_r,
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
    "BacktestMetrics",
    "BacktestResult",
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
    "build_equity_curve_r",
    "calculate_backtest_metrics",
    "calculate_entry_fill",
    "calculate_exit_fill",
    "calculate_gross_r",
    "calculate_maximum_drawdown_r",
    "calculate_position_size",
    "calculate_target_price",
    "close_at_end_of_data",
    "close_position",
    "evaluate_position_on_candle",
    "load_candles_csv",
    "load_signals_csv",
    "open_position",
    "run_backtest",
    "simulate_position",
]

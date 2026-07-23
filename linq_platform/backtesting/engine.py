"""Multi-signal historical backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from linq_platform.backtesting.config import BacktestConfig
from linq_platform.backtesting.execution import open_position
from linq_platform.backtesting.metrics import (
    BacktestMetrics,
    build_equity_curve_r,
    calculate_backtest_metrics,
)
from linq_platform.backtesting.models import (
    Candle,
    Trade,
    TradeSignal,
)
from linq_platform.backtesting.simulator import simulate_position


@dataclass(frozen=True)
class BacktestResult:
    """Complete output from one historical backtest."""

    candles_processed: int
    signals_received: int
    signals_executed: int
    signals_skipped: int
    trades: tuple[Trade, ...]
    metrics: BacktestMetrics
    equity_curve_r: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible backtest result."""

        return {
            "candles_processed": self.candles_processed,
            "signals_received": self.signals_received,
            "signals_executed": self.signals_executed,
            "signals_skipped": self.signals_skipped,
            "metrics": self.metrics.to_dict(),
            "equity_curve_r": list(self.equity_curve_r),
            "trades": [trade.to_dict() for trade in self.trades],
        }


def _validate_candles(
    candles: Sequence[Candle],
) -> None:
    if not candles:
        raise ValueError("candles cannot be empty.")

    previous_time = None

    for candle in candles:
        if previous_time is not None and candle.time <= previous_time:
            raise ValueError(
                "Candles must be strictly ordered by time with no duplicate timestamps."
            )

        previous_time = candle.time


def _signal_entry_indices(
    candles: Sequence[Candle],
) -> dict[object, int]:
    return {candle.time: index for index, candle in enumerate(candles)}


def _open_trade_count_at_index(
    trades: Sequence[Trade],
    candle_index: int,
) -> int:
    return sum(1 for trade in trades if (trade.entry_index <= candle_index < trade.exit_index))


def run_backtest(
    *,
    candles: Sequence[Candle],
    signals: Sequence[TradeSignal],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Simulate all eligible signals across historical candles.

    Signals must share an exact timestamp with a candle. A signal is
    skipped when its timestamp is absent, when it occurs on the final
    candle, or when the maximum-open-position limit has been reached.
    """

    resolved_config = config or BacktestConfig()
    _validate_candles(candles)

    time_to_index = _signal_entry_indices(candles)
    completed_trades: list[Trade] = []
    signals_skipped = 0

    sorted_signals = sorted(
        signals,
        key=lambda signal: (
            signal.time,
            signal.signal_id,
        ),
    )

    for signal in sorted_signals:
        entry_index = time_to_index.get(signal.time)

        if entry_index is None:
            signals_skipped += 1
            continue

        if entry_index >= len(candles) - 1:
            signals_skipped += 1
            continue

        open_count = _open_trade_count_at_index(
            completed_trades,
            entry_index,
        )

        if open_count >= resolved_config.risk.maximum_open_positions:
            signals_skipped += 1
            continue

        position = open_position(
            signal=signal,
            entry_index=entry_index,
            config=resolved_config,
        )

        trade = simulate_position(
            candles=candles,
            position=position,
            config=resolved_config,
        )
        completed_trades.append(trade)

    completed_trades.sort(
        key=lambda trade: (
            trade.entry_index,
            trade.signal_id,
        )
    )

    metrics = calculate_backtest_metrics(completed_trades)
    equity_curve = build_equity_curve_r(completed_trades)

    return BacktestResult(
        candles_processed=len(candles),
        signals_received=len(signals),
        signals_executed=len(completed_trades),
        signals_skipped=signals_skipped,
        trades=tuple(completed_trades),
        metrics=metrics,
        equity_curve_r=tuple(equity_curve),
    )

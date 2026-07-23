"""Performance metrics for completed backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Any, Sequence

from linq_platform.backtesting.models import Trade


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregated performance statistics for simulated trades."""

    trade_count: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    gross_profit_r: float
    gross_loss_r: float
    net_r: float
    average_r: float
    average_win_r: float
    average_loss_r: float
    expectancy_r: float
    profit_factor: float
    maximum_drawdown_r: float
    maximum_consecutive_wins: int
    maximum_consecutive_losses: int

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a serializable dictionary."""

        return asdict(self)


def build_equity_curve_r(
    trades: Sequence[Trade],
) -> list[float]:
    """Return cumulative net R, beginning at zero."""

    equity = 0.0
    curve = [equity]

    for trade in trades:
        equity += trade.net_r
        curve.append(equity)

    return curve


def calculate_maximum_drawdown_r(
    equity_curve: Sequence[float],
) -> float:
    """Calculate the largest peak-to-trough decline in R."""

    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    maximum_drawdown = 0.0

    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        maximum_drawdown = max(
            maximum_drawdown,
            drawdown,
        )

    return float(maximum_drawdown)


def _maximum_streak(
    trades: Sequence[Trade],
    *,
    winning: bool,
) -> int:
    maximum = 0
    current = 0

    for trade in trades:
        condition = trade.net_r > 0

        if not winning:
            condition = trade.net_r < 0

        if condition:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum


def calculate_backtest_metrics(
    trades: Sequence[Trade],
) -> BacktestMetrics:
    """Calculate aggregate metrics from completed trades."""

    trade_count = len(trades)

    if trade_count == 0:
        return BacktestMetrics(
            trade_count=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            loss_rate=0.0,
            gross_profit_r=0.0,
            gross_loss_r=0.0,
            net_r=0.0,
            average_r=0.0,
            average_win_r=0.0,
            average_loss_r=0.0,
            expectancy_r=0.0,
            profit_factor=0.0,
            maximum_drawdown_r=0.0,
            maximum_consecutive_wins=0,
            maximum_consecutive_losses=0,
        )

    wins = [trade.net_r for trade in trades if trade.net_r > 0]
    losses = [trade.net_r for trade in trades if trade.net_r < 0]
    breakeven = [trade.net_r for trade in trades if trade.net_r == 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakeven)

    gross_profit_r = sum(wins)
    gross_loss_r = abs(sum(losses))
    net_r = sum(trade.net_r for trade in trades)

    average_win_r = gross_profit_r / winning_trades if winning_trades else 0.0
    average_loss_r = sum(losses) / losing_trades if losing_trades else 0.0
    average_r = net_r / trade_count

    if gross_loss_r > 0:
        profit_factor = gross_profit_r / gross_loss_r
    elif gross_profit_r > 0:
        profit_factor = inf
    else:
        profit_factor = 0.0

    equity_curve = build_equity_curve_r(trades)

    return BacktestMetrics(
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=winning_trades / trade_count,
        loss_rate=losing_trades / trade_count,
        gross_profit_r=float(gross_profit_r),
        gross_loss_r=float(gross_loss_r),
        net_r=float(net_r),
        average_r=float(average_r),
        average_win_r=float(average_win_r),
        average_loss_r=float(average_loss_r),
        expectancy_r=float(average_r),
        profit_factor=float(profit_factor),
        maximum_drawdown_r=calculate_maximum_drawdown_r(equity_curve),
        maximum_consecutive_wins=_maximum_streak(
            trades,
            winning=True,
        ),
        maximum_consecutive_losses=_maximum_streak(
            trades,
            winning=False,
        ),
    )

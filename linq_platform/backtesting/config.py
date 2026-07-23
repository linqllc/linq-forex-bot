"""Configuration models for the native backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    """Market execution assumptions used during simulation."""

    spread_pips: float = 0.0
    slippage_pips: float = 0.0
    commission_r: float = 0.0
    pip_size: float = 0.0001
    same_bar_exit_policy: str = "stop_first"

    def __post_init__(self) -> None:
        if self.spread_pips < 0:
            raise ValueError("spread_pips cannot be negative.")

        if self.slippage_pips < 0:
            raise ValueError("slippage_pips cannot be negative.")

        if self.commission_r < 0:
            raise ValueError("commission_r cannot be negative.")

        if self.pip_size <= 0:
            raise ValueError("pip_size must be greater than zero.")

        valid_policies = {
            "stop_first",
            "target_first",
        }

        if self.same_bar_exit_policy not in valid_policies:
            raise ValueError(
                "same_bar_exit_policy must be one of: " + ", ".join(sorted(valid_policies))
            )


@dataclass(frozen=True)
class RiskConfig:
    """Risk rules applied to each simulated trade."""

    account_balance: float = 10_000.0
    risk_per_trade: float = 0.01
    maximum_open_positions: int = 1

    def __post_init__(self) -> None:
        if self.account_balance <= 0:
            raise ValueError("account_balance must be greater than zero.")

        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be greater than zero and no greater than one.")

        if self.maximum_open_positions < 1:
            raise ValueError("maximum_open_positions must be at least 1.")


@dataclass(frozen=True)
class StrategyConfig:
    """Core exit and holding rules for a simulated strategy."""

    target_r: float = 2.0
    maximum_holding_bars: int = 24
    allow_long: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.target_r <= 0:
            raise ValueError("target_r must be greater than zero.")

        if self.maximum_holding_bars < 1:
            raise ValueError("maximum_holding_bars must be at least 1.")

        if not self.allow_long and not self.allow_short:
            raise ValueError("At least one trade direction must be enabled.")


@dataclass(frozen=True)
class BacktestConfig:
    """Complete configuration for a backtest run."""

    strategy: StrategyConfig = StrategyConfig()
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()

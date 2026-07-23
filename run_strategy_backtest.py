"""Generate native strategy signals and run a backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from linq_platform.backtesting import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
    load_candles_csv,
    run_backtest,
)
from linq_platform.backtesting.reporting import (
    export_backtest_reports,
)
from linq_platform.strategies import (
    EmaAtrStrategyConfig,
    generate_ema_atr_signals,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EMA/ATR signals from historical candles and run the native backtester."
        )
    )

    parser.add_argument(
        "--candles",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--fast-period",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--slow-period",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--atr-stop-multiplier",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--minimum-stop-pips",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--target-r",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-holding-bars",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--spread-pips",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--commission-r",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pip-size",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--account-balance",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-open-positions",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
    )
    parser.add_argument(
        "--short-only",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.long_only and args.short_only:
        raise ValueError("--long-only and --short-only cannot be used together.")

    candles = load_candles_csv(args.candles)

    strategy_config = EmaAtrStrategyConfig(
        fast_period=args.fast_period,
        slow_period=args.slow_period,
        atr_period=args.atr_period,
        atr_stop_multiplier=(args.atr_stop_multiplier),
        minimum_stop_pips=(args.minimum_stop_pips),
        pip_size=args.pip_size,
        allow_long=not args.short_only,
        allow_short=not args.long_only,
    )

    signals = generate_ema_atr_signals(
        candles,
        strategy_config,
    )

    backtest_config = BacktestConfig(
        strategy=StrategyConfig(
            target_r=args.target_r,
            maximum_holding_bars=(args.max_holding_bars),
        ),
        execution=ExecutionConfig(
            spread_pips=args.spread_pips,
            slippage_pips=args.slippage_pips,
            commission_r=args.commission_r,
            pip_size=args.pip_size,
        ),
        risk=RiskConfig(
            account_balance=args.account_balance,
            risk_per_trade=args.risk_per_trade,
            maximum_open_positions=(args.max_open_positions),
        ),
    )

    result = run_backtest(
        candles=candles,
        signals=signals,
        config=backtest_config,
    )

    reports = export_backtest_reports(
        result,
        Path(args.output),
    )

    print("Strategy backtest complete.")
    print(f"Candles processed: {len(candles)}")
    print(f"Signals generated: {len(signals)}")
    print(f"Signals executed: {result.signals_executed}")
    print(f"Signals skipped: {result.signals_skipped}")
    print(f"Trade count: {result.metrics.trade_count}")
    print(f"Win rate: {result.metrics.win_rate:.2%}")
    print(f"Net R: {result.metrics.net_r:.4f}")
    print(f"Expectancy R: {result.metrics.expectancy_r:.4f}")
    print(f"Profit factor: {result.metrics.profit_factor}")
    print(f"Maximum drawdown R: {result.metrics.maximum_drawdown_r:.4f}")

    for name, path in reports.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

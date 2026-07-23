"""Run the native LINQ historical backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

from linq_platform.backtesting import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
    load_candles_csv,
    load_signals_csv,
    run_backtest,
)
from linq_platform.backtesting.reporting import (
    export_backtest_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run a historical backtest from candle and signal CSV files.")
    )

    parser.add_argument(
        "--candles",
        required=True,
        help="Path to the historical candle CSV.",
    )
    parser.add_argument(
        "--signals",
        required=True,
        help="Path to the trade signal CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for generated reports.",
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
        default=0.0,
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
        "--same-bar-policy",
        choices=[
            "stop_first",
            "target_first",
        ],
        default="stop_first",
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

    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = BacktestConfig(
        strategy=StrategyConfig(
            target_r=args.target_r,
            maximum_holding_bars=(args.max_holding_bars),
        ),
        execution=ExecutionConfig(
            spread_pips=args.spread_pips,
            slippage_pips=args.slippage_pips,
            commission_r=args.commission_r,
            pip_size=args.pip_size,
            same_bar_exit_policy=(args.same_bar_policy),
        ),
        risk=RiskConfig(
            account_balance=args.account_balance,
            risk_per_trade=args.risk_per_trade,
            maximum_open_positions=(args.max_open_positions),
        ),
    )

    candles = load_candles_csv(args.candles)
    signals = load_signals_csv(args.signals)

    result = run_backtest(
        candles=candles,
        signals=signals,
        config=config,
    )

    reports = export_backtest_reports(
        result,
        Path(args.output),
    )

    print("Backtest complete.")
    print(f"Candles processed: {result.candles_processed}")
    print(f"Signals received: {result.signals_received}")
    print(f"Signals executed: {result.signals_executed}")
    print(f"Signals skipped: {result.signals_skipped}")
    print(f"Net R: {result.metrics.net_r:.4f}")
    print(f"Expectancy R: {result.metrics.expectancy_r:.4f}")
    print(f"Maximum drawdown R: {result.metrics.maximum_drawdown_r:.4f}")

    for name, path in reports.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

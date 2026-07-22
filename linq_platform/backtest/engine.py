from __future__ import annotations

import json
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from linq_platform.backtest.strategy import (
    LinqSignalReplayStrategy,
)
from linq_platform.config import BacktestConfig
from linq_platform.data.loader import (
    load_oanda_m5,
    load_selected_signals,
)


def _safe_analysis(analyzer):
    try:
        return analyzer.get_analysis()
    except Exception:
        return {}


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _performance_metrics(
    values: pd.Series,
) -> dict:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "maximum_drawdown_r": 0.0,
            "longest_losing_streak": 0,
        }

    wins = int((values > 0).sum())
    losses = int((values <= 0).sum())

    gross_profit = float(
        values[values > 0].sum()
    )
    gross_loss = float(
        -values[values < 0].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    equity = values.cumsum()
    running_high = equity.cummax()
    drawdown = running_high - equity

    longest_losing_streak = 0
    current_streak = 0

    for value in values:
        if value <= 0:
            current_streak += 1
            longest_losing_streak = max(
                longest_losing_streak,
                current_streak,
            )
        else:
            current_streak = 0

    return {
        "trades": int(len(values)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(values),
        "net_r": float(values.sum()),
        "expectancy_r": float(values.mean()),
        "profit_factor": float(profit_factor),
        "maximum_drawdown_r": float(
            drawdown.max()
        ),
        "longest_losing_streak": int(
            longest_losing_streak
        ),
    }


def _build_parity_audit(
    trade_log: pd.DataFrame,
    output_path: Path,
) -> dict:
    audit = trade_log.copy()

    for column in [
        "phase4_gross_result_r",
        "phase4_net_result_r",
        "realized_r_before_cost",
        "realized_r_after_all_costs",
    ]:
        audit[column] = pd.to_numeric(
            audit.get(column),
            errors="coerce",
        )

    audit["gross_r_difference"] = (
        audit["realized_r_before_cost"]
        - audit["phase4_gross_result_r"]
    )

    audit["net_r_difference"] = (
        audit["realized_r_after_all_costs"]
        - audit["phase4_net_result_r"]
    )

    audit["same_outcome_sign"] = (
        np.sign(audit["realized_r_before_cost"])
        ==
        np.sign(audit["phase4_gross_result_r"])
    )

    audit.to_csv(
        output_path,
        index=False,
    )

    return {
        "matched_trades": int(len(audit)),
        "same_outcome_trades": int(
            audit["same_outcome_sign"].sum()
        ),
        "phase4_gross_r": float(
            audit["phase4_gross_result_r"].sum()
        ),
        "backtrader_gross_r": float(
            audit["realized_r_before_cost"].sum()
        ),
        "gross_parity_difference_r": float(
            audit["gross_r_difference"].sum()
        ),
        "phase4_net_r": float(
            audit["phase4_net_result_r"].sum()
        ),
        "backtrader_net_r": float(
            audit["realized_r_after_all_costs"].sum()
        ),
        "net_parity_difference_r": float(
            audit["net_r_difference"].sum()
        ),
    }


def run_backtest(
    candles_path: str | Path,
    signals_path: str | Path,
    output_dir: str | Path,
    config: BacktestConfig | None = None,
):
    config = config or BacktestConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candles = load_oanda_m5(candles_path)
    signals = load_selected_signals(signals_path)

    if signals.empty:
        raise RuntimeError(
            "No selected signals were loaded."
        )

    first_signal = (
        signals["timestamp"]
        .min()
        .tz_convert("UTC")
        .tz_localize(None)
    )
    last_signal = (
        signals["timestamp"]
        .max()
        .tz_convert("UTC")
        .tz_localize(None)
    )

    start = first_signal - pd.Timedelta(days=7)
    end = last_signal + pd.Timedelta(days=7)

    test_candles = candles.loc[
        (candles.index >= start)
        & (candles.index <= end)
    ].copy()

    if test_candles.empty:
        raise RuntimeError(
            "No candles overlap the selected-signal period."
        )

    cerebro = bt.Cerebro(
        preload=config.preload,
        runonce=config.runonce,
        stdstats=False,
    )

    data = bt.feeds.PandasData(
        dataname=test_candles,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
    )

    cerebro.adddata(
        data,
        name=config.symbol,
    )

    cerebro.addstrategy(
        LinqSignalReplayStrategy,
        signals=signals,
        fixed_units=config.fixed_units,
        pip_size=config.pip_size,
        slippage_pips_each_side=(
            config.slippage_pips_each_side
        ),
        maximum_open_positions=(
            config.maximum_open_positions
        ),
    )

    cerebro.broker.setcash(
        config.starting_cash
    )

    cerebro.broker.set_slippage_fixed(
        fixed=(
            config.slippage_pips_each_side
            * config.pip_size
        ),
        slip_open=True,
        slip_limit=False,
        slip_match=True,
        slip_out=False,
    )

    cerebro.addanalyzer(
        bt.analyzers.TradeAnalyzer,
        _name="trades",
    )
    cerebro.addanalyzer(
        bt.analyzers.DrawDown,
        _name="drawdown",
    )
    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name="returns",
    )
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0.0,
    )

    starting_value = (
        cerebro.broker.getvalue()
    )

    strategies = cerebro.run()
    strategy = strategies[0]

    ending_value = (
        cerebro.broker.getvalue()
    )

    trade_log = pd.DataFrame(
        strategy.trade_log
    )
    signal_log = pd.DataFrame(
        strategy.signal_log
    )
    skipped = pd.DataFrame(
        strategy.skipped_signals
    )

    trade_log_path = (
        output_dir
        / "EUR_USD_v10_trade_ledger.csv"
    )
    signal_log_path = (
        output_dir
        / "EUR_USD_v10_signal_log.csv"
    )
    skipped_path = (
        output_dir
        / "EUR_USD_v10_skipped_signals.csv"
    )
    parity_path = (
        output_dir
        / "EUR_USD_v10_parity_audit.csv"
    )
    summary_path = (
        output_dir
        / "EUR_USD_v10_summary.json"
    )

    trade_log.to_csv(
        trade_log_path,
        index=False,
    )
    signal_log.to_csv(
        signal_log_path,
        index=False,
    )
    skipped.to_csv(
        skipped_path,
        index=False,
    )

    if trade_log.empty:
        gross_metrics = _performance_metrics(
            pd.Series(dtype=float)
        )
        net_metrics = _performance_metrics(
            pd.Series(dtype=float)
        )
        parity = {
            "matched_trades": 0,
            "same_outcome_trades": 0,
            "phase4_gross_r": 0.0,
            "backtrader_gross_r": 0.0,
            "gross_parity_difference_r": 0.0,
            "phase4_net_r": 0.0,
            "backtrader_net_r": 0.0,
            "net_parity_difference_r": 0.0,
        }
        spread_cost_r = 0.0
        slippage_cost_r = 0.0
        total_execution_cost_r = 0.0
    else:
        gross_metrics = _performance_metrics(
            trade_log["realized_r_before_cost"]
        )
        net_metrics = _performance_metrics(
            trade_log[
                "realized_r_after_all_costs"
            ]
        )

        spread_cost_r = float(
            pd.to_numeric(
                trade_log["spread_cost_r"],
                errors="coerce",
            ).fillna(0).sum()
        )
        slippage_cost_r = float(
            pd.to_numeric(
                trade_log["slippage_cost_r"],
                errors="coerce",
            ).fillna(0).sum()
        )
        total_execution_cost_r = float(
            pd.to_numeric(
                trade_log[
                    "total_execution_cost_r"
                ],
                errors="coerce",
            ).fillna(0).sum()
        )

        parity = _build_parity_audit(
            trade_log=trade_log,
            output_path=parity_path,
        )

    summary = {
        "platform": (
            "LINQ Trading Platform V10"
        ),
        "engine": "Backtrader",
        "symbol": config.symbol,
        "candles_loaded": int(
            len(candles)
        ),
        "candles_in_test": int(
            len(test_candles)
        ),
        "signals_loaded": int(
            len(signals)
        ),
        "signals_submitted": int(
            len(signal_log)
        ),
        "signals_skipped": int(
            len(skipped)
        ),
        "gross_performance": gross_metrics,
        "net_performance": net_metrics,
        "spread_cost_r": spread_cost_r,
        "slippage_cost_r": slippage_cost_r,
        "total_execution_cost_r": (
            total_execution_cost_r
        ),
        "starting_value": float(
            starting_value
        ),
        "ending_value": float(
            ending_value
        ),
        "broker_net_change": float(
            ending_value - starting_value
        ),
        "parity": parity,
        "trade_analyzer": _json_safe(
            _safe_analysis(
                strategy.analyzers.trades
            )
        ),
        "drawdown_analyzer": _json_safe(
            _safe_analysis(
                strategy.analyzers.drawdown
            )
        ),
        "returns_analyzer": _json_safe(
            _safe_analysis(
                strategy.analyzers.returns
            )
        ),
        "sharpe_analyzer": _json_safe(
            _safe_analysis(
                strategy.analyzers.sharpe
            )
        ),
        "files": {
            "trade_ledger": str(
                trade_log_path
            ),
            "signal_log": str(
                signal_log_path
            ),
            "skipped_signals": str(
                skipped_path
            ),
            "parity_audit": str(
                parity_path
            ),
            "summary": str(
                summary_path
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )

    return summary


def print_summary(summary):
    gross = summary["gross_performance"]
    net = summary["net_performance"]
    parity = summary["parity"]

    print("=" * 108)
    print(
        "LINQ TRADING PLATFORM V10 — "
        "NATIVE COST ACCOUNTING"
    )
    print("=" * 108)

    print("\nDATA")
    print("-" * 108)
    print(
        f"Candles loaded:                 "
        f"{summary['candles_loaded']:,}"
    )
    print(
        f"Candles in replay period:       "
        f"{summary['candles_in_test']:,}"
    )
    print(
        f"Signals loaded:                 "
        f"{summary['signals_loaded']}"
    )
    print(
        f"Signals submitted:              "
        f"{summary['signals_submitted']}"
    )
    print(
        f"Signals skipped:                "
        f"{summary['signals_skipped']}"
    )

    print("\nNATIVE BACKTRADER RESULTS")
    print("-" * 108)
    print(
        f"Completed trades:               "
        f"{net['trades']}"
    )
    print(
        f"Wins / losses after costs:      "
        f"{net['wins']} / {net['losses']}"
    )
    print(
        f"Win rate after costs:           "
        f"{net['win_rate']:.1%}"
    )
    print(
        f"Gross R:                        "
        f"{gross['net_r']:+.3f}R"
    )
    print(
        f"Spread cost:                    "
        f"-{summary['spread_cost_r']:.3f}R"
    )
    print(
        f"Slippage cost:                  "
        f"-{summary['slippage_cost_r']:.3f}R"
    )
    print(
        f"Total execution cost:           "
        f"-{summary['total_execution_cost_r']:.3f}R"
    )
    print(
        f"Net R after all costs:          "
        f"{net['net_r']:+.3f}R"
    )
    print(
        f"Net expectancy:                 "
        f"{net['expectancy_r']:+.3f}R"
    )
    print(
        f"Net profit factor:              "
        f"{net['profit_factor']:.3f}"
    )
    print(
        f"Maximum drawdown:               "
        f"{net['maximum_drawdown_r']:.3f}R"
    )
    print(
        f"Longest losing streak:          "
        f"{net['longest_losing_streak']}"
    )

    print("\nAUTOMATIC PHASE 4.1 PARITY")
    print("-" * 108)
    print(
        f"Matched trades:                 "
        f"{parity['matched_trades']}"
    )
    print(
        f"Same outcome trades:            "
        f"{parity['same_outcome_trades']}"
        f" / {parity['matched_trades']}"
    )
    print(
        f"Phase 4.1 gross R:              "
        f"{parity['phase4_gross_r']:+.3f}R"
    )
    print(
        f"Backtrader gross R:             "
        f"{parity['backtrader_gross_r']:+.3f}R"
    )
    print(
        f"Gross parity difference:        "
        f"{parity['gross_parity_difference_r']:+.3f}R"
    )
    print(
        f"Phase 4.1 net R:                "
        f"{parity['phase4_net_r']:+.3f}R"
    )
    print(
        f"Backtrader net R:               "
        f"{parity['backtrader_net_r']:+.3f}R"
    )
    print(
        f"Net parity difference:          "
        f"{parity['net_parity_difference_r']:+.3f}R"
    )

    print("\nBROKER ACCOUNT")
    print("-" * 108)
    print(
        f"Starting broker value:          "
        f"${summary['starting_value']:,.2f}"
    )
    print(
        f"Ending broker value:            "
        f"${summary['ending_value']:,.2f}"
    )
    print(
        f"Broker price-P&L change:        "
        f"${summary['broker_net_change']:+,.2f}"
    )
    print(
        "Note: R-based net performance above "
        "includes modeled spread and slippage."
    )

    print("\nFILES")
    print("-" * 108)

    for label, path in summary["files"].items():
        print(f"{label:<32}{path}")

    print("=" * 108)

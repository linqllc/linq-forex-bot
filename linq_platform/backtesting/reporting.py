"""Backtest report exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from linq_platform.backtesting.engine import (
    BacktestResult,
)


def _json_safe(value: Any) -> Any:
    if value == float("inf"):
        return "Infinity"

    if value == float("-inf"):
        return "-Infinity"

    return value


def export_trade_journal_csv(
    result: BacktestResult,
    path: str | Path,
) -> Path:
    """Write completed trades to CSV."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [trade.to_dict() for trade in result.trades]

    fieldnames = [
        "trade_id",
        "signal_id",
        "direction",
        "entry_time",
        "exit_time",
        "entry_index",
        "exit_index",
        "entry_price",
        "exit_price",
        "stop_price",
        "target_price",
        "quantity",
        "risk_amount",
        "gross_r",
        "costs_r",
        "net_r",
        "pnl_amount",
        "bars_held",
        "exit_reason",
        "metadata",
    ]

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            row["metadata"] = json.dumps(
                row.get("metadata"),
                sort_keys=True,
            )
            writer.writerow(row)

    return destination


def export_equity_curve_csv(
    result: BacktestResult,
    path: str | Path,
) -> Path:
    """Write cumulative R equity values to CSV."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "cumulative_r",
            ],
        )
        writer.writeheader()

        for step, cumulative_r in enumerate(result.equity_curve_r):
            writer.writerow(
                {
                    "step": step,
                    "cumulative_r": cumulative_r,
                }
            )

    return destination


def export_metrics_json(
    result: BacktestResult,
    path: str | Path,
) -> Path:
    """Write summary metrics to JSON."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {key: _json_safe(value) for key, value in result.metrics.to_dict().items()}

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination


def export_backtest_json(
    result: BacktestResult,
    path: str | Path,
) -> Path:
    """Write the complete backtest result to JSON."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = result.to_dict()
    payload["metrics"] = {key: _json_safe(value) for key, value in payload["metrics"].items()}

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination


def export_backtest_reports(
    result: BacktestResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Export all standard backtest reports."""

    output = Path(output_directory)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "trades": export_trade_journal_csv(
            result,
            output / "trades.csv",
        ),
        "equity_curve": export_equity_curve_csv(
            result,
            output / "equity_curve.csv",
        ),
        "metrics": export_metrics_json(
            result,
            output / "metrics.json",
        ),
        "backtest": export_backtest_json(
            result,
            output / "backtest.json",
        ),
    }

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from linq_platform.backtesting import (
    load_candles_csv,
    load_signals_csv,
    run_backtest,
)
from linq_platform.backtesting.reporting import (
    export_backtest_reports,
)


def write_candles(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "time,open,high,low,close,volume",
                ("2026-01-01 00:00:00,1.1000,1.1010,1.0990,1.1000,100"),
                ("2026-01-01 01:00:00,1.1000,1.1110,1.0990,1.1100,110"),
                ("2026-01-01 02:00:00,1.1000,1.1010,1.0990,1.1000,120"),
                ("2026-01-01 03:00:00,1.1000,1.1010,1.0890,1.0900,130"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_signals(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                ("signal_id,time,direction,entry_price,stop_price,target_price,order_type"),
                ("long-1,2026-01-01 00:00:00,buy,1.1000,1.0950,,market"),
                ("short-1,2026-01-01 02:00:00,sell,1.1000,1.1050,,market"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_candles_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candles.csv"
    write_candles(path)

    candles = load_candles_csv(path)

    assert len(candles) == 4
    assert candles[0].open == pytest.approx(1.1)
    assert candles[0].volume == pytest.approx(100)


def test_load_signals_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signals.csv"
    write_signals(path)

    signals = load_signals_csv(path)

    assert len(signals) == 2
    assert signals[0].direction.value == "long"
    assert signals[1].direction.value == "short"


def test_missing_candle_column_raises(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candles.csv"
    path.write_text(
        "time,open,low,close\n2026-01-01 00:00:00,1.1,1.0,1.05\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        load_candles_csv(path)


def test_invalid_signal_direction_raises(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signals.csv"
    path.write_text(
        ("signal_id,time,direction,entry_price,stop_price\n")
        + ("bad,2026-01-01 00:00:00,sideways,1.1,1.0\n"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="direction",
    ):
        load_signals_csv(path)


def test_export_backtest_reports(
    tmp_path: Path,
) -> None:
    candle_path = tmp_path / "candles.csv"
    signal_path = tmp_path / "signals.csv"
    output_path = tmp_path / "reports"

    write_candles(candle_path)
    write_signals(signal_path)

    result = run_backtest(
        candles=load_candles_csv(candle_path),
        signals=load_signals_csv(signal_path),
    )

    reports = export_backtest_reports(
        result,
        output_path,
    )

    assert reports["trades"].exists()
    assert reports["equity_curve"].exists()
    assert reports["metrics"].exists()
    assert reports["backtest"].exists()

    metrics = json.loads(reports["metrics"].read_text(encoding="utf-8"))

    assert metrics["trade_count"] == 2
    assert metrics["net_r"] == pytest.approx(4.0)

    with reports["trades"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["signal_id"] == "long-1"


def test_cli_end_to_end(
    tmp_path: Path,
) -> None:
    candle_path = tmp_path / "candles.csv"
    signal_path = tmp_path / "signals.csv"
    output_path = tmp_path / "output"

    write_candles(candle_path)
    write_signals(signal_path)

    completed = subprocess.run(
        [
            sys.executable,
            "run_backtest.py",
            "--candles",
            str(candle_path),
            "--signals",
            str(signal_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Backtest complete." in completed.stdout
    assert (output_path / "trades.csv").exists()
    assert (output_path / "metrics.json").exists()
    assert (output_path / "backtest.json").exists()

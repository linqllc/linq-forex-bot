from __future__ import annotations

import pandas as pd
import pytest

from src.research_v10.config import ResearchConfig
from src.research_v10.execution_replay import (
    ExecutionConfig,
    PendingOrder,
    RealisticExecutionReplay,
)


def signal(direction: str = "long") -> dict:
    return {
        "setup_id": (
            f"EUR_USD-{direction}-"
            f"2026-01-01T06:00:00+00:00"
        ),
        "instrument": "EUR_USD",
        "direction": direction,
        "impulse_index": 100,
        "entry_index": 101,
        "impulse_timestamp": pd.Timestamp(
            "2026-01-01T05:55:00Z"
        ),
        "timestamp": pd.Timestamp(
            "2026-01-01T06:00:00Z"
        ),
        "entry_price": 1.1000,
        "atr": 0.0010,
        "session": "london_open",
        "retrace_fraction": 0.25,
    }


def candle(
    open_: float,
    high: float,
    low: float,
    close: float,
) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp(
                "2026-01-01T06:05:00Z"
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_long_entry_pays_ask_and_slippage():
    engine = RealisticExecutionReplay(
        ResearchConfig(),
        ExecutionConfig(
            spread_pips=1.0,
            entry_slippage_pips=0.2,
        ),
    )

    trade, error = engine._create_trade(
        PendingOrder(
            signal=signal("long"),
            signal_index=101,
            execution_index=102,
        ),
        candle(
            1.1000,
            1.1010,
            1.0990,
            1.1005,
        ),
        102,
    )

    assert error is None
    assert trade is not None
    assert trade.filled_entry_price == pytest.approx(
        1.10007
    )


def test_short_entry_sells_bid_with_slippage():
    engine = RealisticExecutionReplay(
        ResearchConfig(),
        ExecutionConfig(
            spread_pips=1.0,
            entry_slippage_pips=0.2,
        ),
    )

    trade, error = engine._create_trade(
        PendingOrder(
            signal=signal("short"),
            signal_index=101,
            execution_index=102,
        ),
        candle(
            1.1000,
            1.1010,
            1.0990,
            1.1005,
        ),
        102,
    )

    assert error is None
    assert trade is not None
    assert trade.filled_entry_price == pytest.approx(
        1.09993
    )


def test_large_entry_gap_is_rejected():
    engine = RealisticExecutionReplay(
        ResearchConfig(),
        ExecutionConfig(
            reject_if_entry_gap_atr=0.50,
        ),
    )

    trade, error = engine._create_trade(
        PendingOrder(
            signal=signal("long"),
            signal_index=101,
            execution_index=102,
        ),
        candle(
            1.1010,
            1.1020,
            1.1005,
            1.1015,
        ),
        102,
    )

    assert trade is None
    assert error == "entry_gap_exceeded"


def test_execution_config_rejects_negative_spread():
    with pytest.raises(ValueError):
        RealisticExecutionReplay(
            ResearchConfig(),
            ExecutionConfig(
                spread_pips=-1.0,
            ),
        )

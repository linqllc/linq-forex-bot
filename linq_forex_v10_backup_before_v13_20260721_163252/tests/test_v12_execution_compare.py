from __future__ import annotations

import pandas as pd
import pytest

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.execution_compare import (
    ExactSignalExecutionComparison,
    ExecutionScenario,
)


def make_featured() -> pd.DataFrame:
    rows = []

    for index in range(10):
        rows.append(
            {
                "timestamp": pd.Timestamp(
                    "2026-01-01T06:00:00Z"
                )
                + pd.Timedelta(
                    minutes=5 * index
                ),
                "open": 1.1000,
                "high": 1.1020,
                "low": 1.0980,
                "close": 1.1005,
            }
        )

    return pd.DataFrame(rows)


def make_signal(
    direction: str = "long",
) -> dict:
    return {
        "setup_id": (
            f"EUR_USD-{direction}-test"
        ),
        "instrument": "EUR_USD",
        "direction": direction,
        "impulse_timestamp": pd.Timestamp(
            "2026-01-01T06:00:00Z"
        ),
        "timestamp": pd.Timestamp(
            "2026-01-01T06:05:00Z"
        ),
        "entry_index": 1,
        "entry_price": 1.1000,
        "atr": 0.0010,
        "session": "london_open",
        "retrace_fraction": 0.20,
        "accepted": True,
    }


def test_signal_price_execution_uses_same_bar_timestamp():
    engine = ExactSignalExecutionComparison(
        ResearchConfig(),
        expected_accepted_signals=None,
    )

    scenario = ExecutionScenario(
        name="ideal",
        entry_mode="signal_price",
    )

    trade = engine._create_trade(
        make_signal("long"),
        make_featured(),
        scenario,
    )

    assert trade is not None
    assert trade.entry_index == 1
    assert trade.outcome_start_index == 2
    assert trade.filled_entry_price == pytest.approx(
        1.1000
    )


def test_long_entry_costs_are_adverse():
    engine = ExactSignalExecutionComparison(
        ResearchConfig(),
        expected_accepted_signals=None,
    )

    scenario = ExecutionScenario(
        name="cost",
        entry_mode="signal_price",
        spread_pips=0.8,
        entry_slippage_pips=0.2,
        latency_pips=0.1,
    )

    trade = engine._create_trade(
        make_signal("long"),
        make_featured(),
        scenario,
    )

    assert trade is not None

    assert trade.filled_entry_price == pytest.approx(
        1.10007
    )


def test_short_entry_costs_are_adverse():
    engine = ExactSignalExecutionComparison(
        ResearchConfig(),
        expected_accepted_signals=None,
    )

    scenario = ExecutionScenario(
        name="cost",
        entry_mode="signal_price",
        spread_pips=0.8,
        entry_slippage_pips=0.2,
        latency_pips=0.1,
    )

    trade = engine._create_trade(
        make_signal("short"),
        make_featured(),
        scenario,
    )

    assert trade is not None

    assert trade.filled_entry_price == pytest.approx(
        1.09993
    )


def test_next_open_begins_outcome_after_entry_bar():
    engine = ExactSignalExecutionComparison(
        ResearchConfig(),
        expected_accepted_signals=None,
    )

    scenario = ExecutionScenario(
        name="next_open",
        entry_mode="next_open",
    )

    trade = engine._create_trade(
        make_signal("long"),
        make_featured(),
        scenario,
    )

    assert trade is not None
    assert trade.entry_index == 2
    assert trade.outcome_start_index == 3


def test_identity_hash_is_order_independent():
    engine = ExactSignalExecutionComparison(
        ResearchConfig(),
        expected_accepted_signals=None,
    )

    first = [
        {"setup_id": "b"},
        {"setup_id": "a"},
    ]

    second = [
        {"setup_id": "a"},
        {"setup_id": "b"},
    ]

    assert (
        engine._signal_identity_hash(first)
        == engine._signal_identity_hash(second)
    )

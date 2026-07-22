from __future__ import annotations

import pandas as pd
import pytest

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.intrabar_reconstruction import (
    CausalIntrabarReconstruction,
    IntrabarScenario,
    M5Impulse,
)


def make_impulse(
    direction: str = "long",
) -> M5Impulse:
    return M5Impulse(
        setup_id=f"test-{direction}",
        direction=direction,
        m5_index=1,
        m5_timestamp=pd.Timestamp(
            "2026-01-01T06:00:00Z"
        ),
        available_from=pd.Timestamp(
            "2026-01-01T06:05:00Z"
        ),
        impulse_open=1.1000,
        impulse_high=1.1020,
        impulse_low=1.1000,
        impulse_close=1.1018,
        impulse_range=0.0020,
        atr=0.0010,
        session="london_open",
    )


def make_m1(
    timestamp: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp(
                timestamp
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_m5_open_timestamp_available_after_five_minutes():
    engine = CausalIntrabarReconstruction(
        ResearchConfig(),
        timestamps_are_bar_open=True,
    )

    available = engine._available_from(
        pd.Timestamp(
            "2026-01-01T06:00:00Z"
        ),
        minutes=5,
    )

    assert available == pd.Timestamp(
        "2026-01-01T06:05:00Z"
    )


def test_long_25_percent_retrace_price():
    engine = CausalIntrabarReconstruction(
        ResearchConfig()
    )

    impulse = make_impulse("long")

    fraction = engine._retrace_fraction(
        impulse,
        1.1015,
    )

    assert fraction == pytest.approx(0.25)


def test_short_25_percent_retrace_price():
    engine = CausalIntrabarReconstruction(
        ResearchConfig()
    )

    impulse = make_impulse("short")
    impulse.impulse_open = 1.1020
    impulse.impulse_close = 1.1002

    fraction = engine._retrace_fraction(
        impulse,
        1.1005,
    )

    assert fraction == pytest.approx(0.25)


def test_long_invalidates_below_impulse_low():
    engine = CausalIntrabarReconstruction(
        ResearchConfig()
    )

    assert engine._invalidated(
        make_impulse("long"),
        make_m1(
            "2026-01-01T06:05:00Z",
            1.1002,
            1.1003,
            1.0999,
            1.1001,
        ),
    )


def test_limit_fill_never_uses_pre_close_m1():
    engine = CausalIntrabarReconstruction(
        ResearchConfig()
    )

    impulse = make_impulse("long")

    m1 = pd.DataFrame(
        [
            make_m1(
                "2026-01-01T06:04:00Z",
                1.1016,
                1.1017,
                1.1014,
                1.1015,
            ),
            make_m1(
                "2026-01-01T06:05:00Z",
                1.1016,
                1.1017,
                1.1014,
                1.1015,
            ),
        ]
    )

    window = engine._m1_window(
        impulse,
        m1,
        max_wait_bars=10,
    )

    assert pd.Timestamp(
        window.iloc[0]["timestamp"]
    ) >= impulse.available_from


def test_reversal_entry_starts_outcome_next_m1_bar():
    engine = CausalIntrabarReconstruction(
        ResearchConfig()
    )

    impulse = make_impulse("long")

    window = pd.DataFrame(
        [
            make_m1(
                "2026-01-01T06:05:00Z",
                1.1019,
                1.1019,
                1.1017,
                1.10175,
            ),
            make_m1(
                "2026-01-01T06:06:00Z",
                1.10175,
                1.1019,
                1.1017,
                1.10188,
            ),
        ],
        index=[100, 101],
    )

    entry = engine._reversal_entry(
        impulse,
        window,
        IntrabarScenario(
            name="test",
            entry_mode="reversal_confirm",
            max_retrace_fraction=0.25,
            spread_pips=0.0,
            entry_slippage_pips=0.0,
        ),
    )

    assert entry is not None
    assert entry["entry_m1_index"] == 101
    assert entry["outcome_start_index"] == 102

from __future__ import annotations

import pandas as pd
import pytest

from src.research_v10.config import (
    ResearchConfig,
)
from src.research_v10.pending_limit_replay import (
    PendingLimitConfig,
    PendingLimitReplay,
)


def featured_candle(
    timestamp: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    atr: float = 0.0010,
    range_atr: float = 2.0,
    body_ratio: float = 0.8,
    session: str = "london_open",
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
            "atr": atr,
            "range_atr": range_atr,
            "body_ratio": body_ratio,
            "session": session,
        }
    )


def test_long_impulse_creates_limit_at_25_percent():
    engine = PendingLimitReplay(
        ResearchConfig(),
        PendingLimitConfig(
            retrace_fraction=0.25,
            spread_pips=0.8,
        ),
    )

    order = engine._create_pending_order(
        100,
        featured_candle(
            "2026-01-01T06:00:00Z",
            1.1000,
            1.1020,
            1.1000,
            1.1018,
        ),
    )

    assert order is not None
    assert order.direction == "long"

    assert order.mid_limit_price == pytest.approx(
        1.1015
    )

    assert (
        order.executable_limit_price
        == pytest.approx(1.10154)
    )


def test_short_impulse_creates_limit_at_25_percent():
    engine = PendingLimitReplay(
        ResearchConfig(),
        PendingLimitConfig(
            retrace_fraction=0.25,
            spread_pips=0.8,
        ),
    )

    order = engine._create_pending_order(
        100,
        featured_candle(
            "2026-01-01T06:00:00Z",
            1.1020,
            1.1020,
            1.1000,
            1.1002,
        ),
    )

    assert order is not None
    assert order.direction == "short"

    assert order.mid_limit_price == pytest.approx(
        1.1005
    )

    assert (
        order.executable_limit_price
        == pytest.approx(1.10046)
    )


def test_long_limit_fills_when_ask_low_touches():
    engine = PendingLimitReplay(
        ResearchConfig(),
        PendingLimitConfig(
            retrace_fraction=0.25,
            spread_pips=0.8,
        ),
    )

    order = engine._create_pending_order(
        100,
        featured_candle(
            "2026-01-01T06:00:00Z",
            1.1000,
            1.1020,
            1.1000,
            1.1018,
        ),
    )

    assert order is not None

    fill_candle = featured_candle(
        "2026-01-01T06:05:00Z",
        1.1017,
        1.1018,
        1.1015,
        1.1016,
        range_atr=0.3,
        body_ratio=0.4,
    )

    assert engine._order_touched(
        order,
        fill_candle,
    )


def test_long_order_invalidates_below_impulse_low():
    engine = PendingLimitReplay(
        ResearchConfig(),
        PendingLimitConfig(),
    )

    order = engine._create_pending_order(
        100,
        featured_candle(
            "2026-01-01T06:00:00Z",
            1.1000,
            1.1020,
            1.1000,
            1.1018,
        ),
    )

    assert order is not None

    invalidation_candle = featured_candle(
        "2026-01-01T06:05:00Z",
        1.1001,
        1.1003,
        1.0999,
        1.1000,
        range_atr=0.4,
        body_ratio=0.2,
    )

    assert engine._order_invalidated(
        order,
        invalidation_candle,
    )


def test_negative_spread_is_rejected():
    with pytest.raises(ValueError):
        PendingLimitReplay(
            ResearchConfig(),
            PendingLimitConfig(
                spread_pips=-0.1,
            ),
        )

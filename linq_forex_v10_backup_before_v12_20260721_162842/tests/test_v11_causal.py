from __future__ import annotations

import pandas as pd

from src.research_v10.config import ResearchConfig
from src.research_v10.signal_engine import CausalSignalEngine


def candle(
    timestamp: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    atr: float = 1.0,
    range_atr: float = 1.5,
    body_ratio: float = 0.8,
    session: str = "london_open",
) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp(timestamp),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "range_atr": range_atr,
            "body_ratio": body_ratio,
            "ema_fast": close,
            "ema_slow": close - 0.1,
            "ema_slow_slope_12": 0.1,
            "ema_distance_atr": 0.1,
            "return_12": 0.01,
            "return_48": 0.02,
            "realized_vol_48": 0.01,
            "hour_utc": 6.25,
            "weekday": 1,
            "session": session,
            "distance_prev_high_atr": 1.0,
            "distance_prev_low_atr": 1.0,
            "bull_fvg_atr": 0.0,
            "bear_fvg_atr": 0.0,
        }
    )


def test_impulse_does_not_signal_on_same_bar():
    engine = CausalSignalEngine(
        ResearchConfig(),
        retrace_threshold=0.25,
        required_session="london_open",
    )

    signals = engine.process_bar(
        100,
        candle(
            "2026-01-01T06:00:00Z",
            100.0,
            102.0,
            100.0,
            101.8,
        ),
    )

    assert signals == []
    assert len(engine.active_impulses) == 1


def test_next_bar_can_emit_causal_signal():
    engine = CausalSignalEngine(
        ResearchConfig(),
        retrace_threshold=0.25,
        required_session="london_open",
    )

    engine.process_bar(
        100,
        candle(
            "2026-01-01T06:00:00Z",
            100.0,
            102.0,
            100.0,
            101.8,
        ),
    )

    signals = engine.process_bar(
        101,
        candle(
            "2026-01-01T06:05:00Z",
            101.7,
            101.8,
            101.5,
            101.5,
            range_atr=0.3,
            body_ratio=0.5,
        ),
    )

    assert len(signals) == 1
    assert signals[0]["entry_index"] == 101
    assert signals[0]["impulse_index"] == 100
    assert signals[0]["accepted"] is True


def test_off_session_signal_is_rejected():
    engine = CausalSignalEngine(
        ResearchConfig(),
        retrace_threshold=0.25,
        required_session="london_open",
    )

    engine.process_bar(
        100,
        candle(
            "2026-01-01T06:00:00Z",
            100.0,
            102.0,
            100.0,
            101.8,
        ),
    )

    signals = engine.process_bar(
        101,
        candle(
            "2026-01-01T12:05:00Z",
            101.7,
            101.8,
            101.5,
            101.5,
            range_atr=0.3,
            body_ratio=0.5,
            session="new_york_overlap",
        ),
    )

    assert len(signals) == 1
    assert signals[0]["accepted"] is False
    assert signals[0]["rejection_reason"] == "locked_rule_failed"


def test_invalidated_impulse_produces_no_signal():
    engine = CausalSignalEngine(
        ResearchConfig(),
    )

    engine.process_bar(
        100,
        candle(
            "2026-01-01T06:00:00Z",
            100.0,
            102.0,
            100.0,
            101.8,
        ),
    )

    signals = engine.process_bar(
        101,
        candle(
            "2026-01-01T06:05:00Z",
            100.2,
            100.4,
            99.9,
            100.1,
            range_atr=0.5,
            body_ratio=0.2,
        ),
    )

    assert signals == []
    assert len(engine.active_impulses) == 0

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_v10.config import ResearchConfig
from src.research_v10.features import add_market_features
from src.research_v10.labels import label_outcomes
from src.research_v10.research import chronological_split, performance_summary
from src.research_v10.setups import detect_candidate_setups


def make_candles(rows: int = 420) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    base = 1.10 + np.arange(rows) * 0.00001
    close = base + np.sin(np.arange(rows) / 10) * 0.0002
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.00008
    low = np.minimum(open_, close) - 0.00008

    # Force a large bullish displacement after warm-up.
    idx = 275
    open_[idx] = close[idx - 1]
    close[idx] = open_[idx] + 0.0015
    high[idx] = close[idx] + 0.0001
    low[idx] = open_[idx] - 0.00005

    # Force pullback into 25-75% impulse.
    open_[idx + 1] = close[idx]
    close[idx + 1] = close[idx] - 0.00065
    high[idx + 1] = open_[idx + 1] + 0.00005
    low[idx + 1] = close[idx + 1] - 0.00005

    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    })


def test_feature_and_setup_pipeline():
    cfg = ResearchConfig(ema_slow=50, max_holding_bars=40)
    candles = add_market_features(make_candles(), 14, 20, 50)
    setups = detect_candidate_setups(candles, cfg)
    assert not setups.empty
    assert {"direction", "entry_price", "impulse_atr", "session"}.issubset(setups.columns)


def test_outcome_labels_have_r_values():
    cfg = ResearchConfig(ema_slow=50, max_holding_bars=40)
    candles = add_market_features(make_candles(), 14, 20, 50)
    setups = detect_candidate_setups(candles, cfg)
    labeled = label_outcomes(candles, setups, cfg)
    assert "r_multiple" in labeled
    assert labeled["r_multiple"].notna().any()


def test_chronological_split_preserves_order():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, tz="UTC"),
        "r_multiple": [1, -1] * 5,
    })
    train, test = chronological_split(df, 0.7)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_performance_summary():
    df = pd.DataFrame({"r_multiple": [1.5, -1.0, 1.5, -1.0]})
    summary = performance_summary(df)
    assert summary["trades"] == 4
    assert summary["expectancy_r"] == 0.25

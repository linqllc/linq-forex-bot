from __future__ import annotations
import numpy as np
import pandas as pd
from supply_demand_v13.config import Config
from supply_demand_v13.data import load_candles
from supply_demand_v13.features import add_features
from supply_demand_v13.engine import _overlap_fraction, _slowing
from supply_demand_v13.backtest import summarize


def synthetic(n=1000):
    rng = np.random.default_rng(7)
    ret = rng.normal(0, 0.00025, n)
    close = 1.10 + np.cumsum(ret)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.00005, 0.00020, n)
    low = np.minimum(open_, close) - rng.uniform(0.00005, 0.00020, n)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def test_overlap():
    assert _overlap_fraction(10, 8, 9, 7) == 0.5


def test_feature_causality_shape():
    out = add_features(synthetic(), Config())
    assert len(out) == 1000
    assert "h1_direction" in out
    assert out["time"].is_monotonic_increasing


def test_slowdown_allows_small_bump():
    df = synthetic(30)
    # Force bearish bodies: earlier 10,8,7; recent 8,6 arbitrary units.
    vals = [10, 8, 7, 8, 6]
    for k, body in enumerate(vals):
        i = 20 - 5 + k
        df.loc[i, "open"] = 1.2
        df.loc[i, "close"] = 1.2 - body * 1e-5
    df["body"] = (df.close-df.open).abs()
    ok, stats = _slowing(df, 20, 1, Config())
    assert ok
    assert stats["recent_avg_ratio"] <= 1.05


def test_summary():
    trades = pd.DataFrame({"r_multiple": [1.5, -1.0, 1.5]})
    s = summarize(trades)
    assert s["trades"] == 3
    assert s["profit_factor"] == 3.0

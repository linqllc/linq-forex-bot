
from pathlib import Path
import numpy as np
import pandas as pd

from run_simple_strategy_discovery import Config, add_features, load_candles, metrics, StrategySpec, simulate

def test_smoke(tmp_path: Path):
    n = 2000
    t = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    close = 1.10 + np.cumsum(rng.normal(0, 0.00015, n))
    df = pd.DataFrame({
        "timestamp": t,
        "open": np.r_[close[0], close[:-1]],
        "high": close + 0.0002,
        "low": close - 0.0002,
        "close": close,
    })
    path = tmp_path / "candles.csv"
    df.to_csv(path, index=False)
    loaded = add_features(load_candles(path))
    spec = StrategySpec("ema_cross_20_50", "both", "ema200", "all", "none", 1.5, 2.0, 48)
    trades = simulate(loaded, spec, Config(min_train_trades=1, min_validation_trades=1, min_test_trades=1))
    result = metrics(trades)
    assert "expectancy_r" in result

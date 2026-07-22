import pandas as pd
from src.features import build_feature_dataset
from src.research import target_probability_table, condition_breakdown


def test_feature_dataset_and_probability_table():
    candles = pd.DataFrame([{ 
        "atr": 0.001, "ema_fast": 1.1, "opening_range_low": 1.09,
        "opening_range_high": 1.11, "mid_close": 1.10
    }])
    setups = pd.DataFrame([{
        "instrument":"EUR_USD", "entry_time":"2026-01-05T10:00:00+00:00",
        "session_date":"2026-01-05", "direction":"long", "entry_index":0,
        "entry":1.101, "zone_low":1.099, "zone_high":1.100,
        "score":90, "trend_aligned":True, "bos_confirmed":True,
        "breakout_body_multiple":2.2, "displacement_atr":2.5,
        "bars_to_retest":4, "max_favorable_r":3.2, "stopped":False, "final_r":2.0
    }])
    features = build_feature_dataset(setups, candles, [1.0, 3.0, 4.0])
    assert features.loc[0, "hit_3.00R"] == 1
    assert features.loc[0, "hit_4.00R"] == 0
    table = target_probability_table(features, [1.0, 3.0, 4.0])
    assert table.loc[table.ratio == 3.0, "hit_rate"].iloc[0] == 1.0
    breakdown = condition_breakdown(features, 3.0)
    assert not breakdown.empty

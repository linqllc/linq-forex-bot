import numpy as np
import pandas as pd
from src.ml_probability import train_probability_models, choose_dynamic_target


def test_probability_engine_handles_small_dataset():
    rows=[]
    for i in range(20):
        rows.append({
            "entry_time":f"2026-01-{(i%20)+1:02d}T10:00:00Z", "score":70+i%5,
            "trend_aligned":i%2,"bos_confirmed":i%3==0,"breakout_body_multiple":1.5,
            "displacement_atr":1.2,"bars_to_retest":2,"zone_width_atr":0.7,
            "ema_distance_atr":0.3,"opening_range_position":1.1,"hour_utc":15,"weekday":i%5,
            "trend_engine_score":70,"structure_engine_score":65,"supply_demand_engine_score":75,
            "liquidity_engine_score":35,"volatility_engine_score":80,"session_engine_score":100,
            "confluence_score":72,"ema_separation_atr":0.2,"volatility_ratio":1.1,
            "liquidity_sweep":0,"choch_confirmed":0,"hit_2.00R":i%4==0,
        })
    frame=pd.DataFrame(rows)
    predictions,metrics,coefs=train_probability_models(frame,[2.0])
    assert "pred_hit_2.00R" in predictions
    assert not metrics.empty
    selected=choose_dynamic_target(predictions,[2.0])
    assert "model_selected_ratio" in selected

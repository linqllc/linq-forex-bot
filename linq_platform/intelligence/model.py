from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAKAGE = {
    "timestamp","entry_price","stop_price","target_price","risk_price",
    "mfe_r","mae_r","stop_hit","valid_risk","bars_observed","result_r",
    "gross_result_r","net_result_r","exit_reason","bars_held","exit_price",
    "spread_cost_r","slippage_cost_r","total_cost_r","selected",
    "probability_1r","actual_win","holdout"
}

def clean_features(row):
    out = {}
    for k, v in row.items():
        s = str(k)
        if k in LEAKAGE or s.startswith("hit_") or s.startswith("bars_to_") or s.startswith("probability"):
            continue
        out[k] = v
    return out

def feature_columns(df):
    excluded = LEAKAGE | {"setup_id","strategy"}
    return [c for c in df.columns if c not in excluded]

def make_model(X):
    numeric = X.select_dtypes(include=["number","bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    tr = []
    if numeric:
        tr.append(("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric))
    if categorical:
        tr.append(("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))]), categorical))
    return Pipeline([("prep",ColumnTransformer(tr)),("model",LogisticRegression(max_iter=3000,class_weight="balanced",C=0.25,random_state=42))])

def predict_holdout(dataset, holdout_start, cfg):
    data = dataset.copy()
    cols = feature_columns(data)
    X = data[cols].copy()
    y = data["actual_win"].astype(int)
    probs = np.full(len(data), np.nan)
    hold = np.zeros(len(data), dtype=bool)
    hold[holdout_start:] = True
    for i in range(holdout_start, len(data)):
        if i < cfg.minimum_training_rows:
            continue
        train_y = y.iloc[:i]
        if train_y.nunique() < 2:
            probs[i] = float(train_y.mean())
        else:
            model = make_model(X.iloc[:i])
            model.fit(X.iloc[:i], train_y)
            probs[i] = model.predict_proba(X.iloc[i:i+1])[0,1]
    data["holdout"] = hold
    data["probability_1r"] = probs
    data["selected"] = data["holdout"] & data["probability_1r"].notna() & (data["probability_1r"] >= cfg.probability_threshold)
    return data

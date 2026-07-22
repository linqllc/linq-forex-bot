from __future__ import annotations

import math
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "score", "trend_aligned", "bos_confirmed", "breakout_body_multiple",
    "displacement_atr", "bars_to_retest", "zone_width_atr",
    "ema_distance_atr", "opening_range_position", "hour_utc", "weekday",
    "trend_engine_score", "structure_engine_score", "supply_demand_engine_score",
    "liquidity_engine_score", "volatility_engine_score", "session_engine_score",
    "confluence_score", "ema_separation_atr", "volatility_ratio",
    "liquidity_sweep", "choch_confirmed",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for a in pos:
        wins += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(wins / total)


def _fit_logistic(X: np.ndarray, y: np.ndarray, iterations: int = 1200, lr: float = 0.05, l2: float = 0.02) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(iterations):
        p = _sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(1, len(y))
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def train_probability_models(features: pd.DataFrame, ratios: list[float], train_fraction: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological logistic models. Returns predictions, metrics, coefficients.

    This is an experimental research layer. It refuses to train on tiny or single-class samples.
    """
    if features.empty:
        return features.copy(), pd.DataFrame(), pd.DataFrame()
    frame = features.sort_values("entry_time").reset_index(drop=True).copy()
    available = [c for c in FEATURE_COLUMNS if c in frame.columns]
    Xraw = frame[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).to_numpy()
    split = max(1, min(len(frame) - 1, int(len(frame) * train_fraction)))
    train_x = Xraw[:split]
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    X = (Xraw - mean) / std
    X = np.column_stack([np.ones(len(X)), X])

    metrics, coefficients = [], []
    for ratio in ratios:
        target = f"hit_{ratio:.2f}R"
        if target not in frame:
            continue
        y = frame[target].astype(float).to_numpy()
        y_train = y[:split]
        status = "trained"
        if len(y_train) < 30 or len(np.unique(y_train)) < 2:
            baseline = float(y_train.mean()) if len(y_train) else 0.0
            probs = np.repeat(baseline, len(frame))
            status = "insufficient_training_data"
            w = np.zeros(X.shape[1])
            w[0] = math.log((baseline + 1e-6) / (1.0 - baseline + 1e-6)) if 0 < baseline < 1 else 0.0
        else:
            w = _fit_logistic(X[:split], y_train)
            probs = _sigmoid(X @ w)
        frame[f"pred_hit_{ratio:.2f}R"] = probs
        test_y, test_p = y[split:], probs[split:]
        brier = float(np.mean((test_p - test_y) ** 2)) if len(test_y) else float("nan")
        metrics.append({
            "ratio": ratio,
            "status": status,
            "train_rows": split,
            "test_rows": len(frame) - split,
            "train_hit_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "test_hit_rate": float(test_y.mean()) if len(test_y) else float("nan"),
            "test_brier_score": brier,
            "test_auc": _auc(test_y, test_p) if len(test_y) else float("nan"),
        })
        for name, coef in zip(["intercept"] + available, w):
            coefficients.append({"ratio": ratio, "feature": name, "coefficient": float(coef), "status": status})
    return frame, pd.DataFrame(metrics), pd.DataFrame(coefficients)


def choose_dynamic_target(predictions: pd.DataFrame, ratios: list[float]) -> pd.DataFrame:
    out = predictions.copy()
    choices, expected_values = [], []
    for row in out.to_dict("records"):
        candidates = []
        for ratio in ratios:
            p = float(row.get(f"pred_hit_{ratio:.2f}R", 0.0))
            ev = p * ratio - (1.0 - p)
            candidates.append((ev, ratio))
        ev, ratio = max(candidates) if candidates else (0.0, 0.0)
        choices.append(ratio)
        expected_values.append(ev)
    out["model_selected_ratio"] = choices
    out["model_expected_value_r"] = expected_values
    return out

from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

from .config import ResearchConfig


NUMERIC_FEATURES = [
    "impulse_atr",
    "impulse_body_ratio",
    "retrace_fraction",
    "pullback_bars",
    "ema_slow_slope_12",
    "ema_distance_atr",
    "return_12",
    "return_48",
    "realized_vol_48",
    "hour_utc",
    "distance_prev_high_atr",
    "distance_prev_low_atr",
    "bull_fvg_atr",
    "bear_fvg_atr",
]


def performance_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0, "win_rate": np.nan, "expectancy_r": np.nan,
            "profit_factor": np.nan, "max_drawdown_r": np.nan,
        }

    r = df["r_multiple"].dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    equity = r.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        "max_drawdown_r": float(drawdown.min()) if not drawdown.empty else 0.0,
        "total_r": float(r.sum()),
    }


def chronological_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def rank_single_features(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for feature in NUMERIC_FEATURES:
        clean = train[[feature, "r_multiple"]].dropna()
        if clean[feature].nunique() < 4:
            continue
        quantiles = clean[feature].quantile([0.25, 0.5, 0.75]).drop_duplicates()
        for q, threshold in quantiles.items():
            for op in ("ge", "le"):
                subset = clean[clean[feature] >= threshold] if op == "ge" else clean[clean[feature] <= threshold]
                if len(subset) < 10:
                    continue
                perf = performance_summary(subset)
                rows.append({
                    "feature": feature,
                    "operator": op,
                    "threshold": float(threshold),
                    "quantile": float(q),
                    **perf,
                })

    for feature in ("session", "weekday", "direction", "trend_aligned"):
        if feature not in train:
            continue
        for value, subset in train.groupby(feature, dropna=False):
            if len(subset) < 8:
                continue
            rows.append({
                "feature": feature,
                "operator": "eq",
                "threshold": value,
                "quantile": np.nan,
                **performance_summary(subset),
            })

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    return ranked.sort_values(["expectancy_r", "trades"], ascending=[False, False]).reset_index(drop=True)


def _apply_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    feature = rule["feature"]
    op = rule["operator"]
    threshold = rule["threshold"]
    if op == "ge":
        return df[feature] >= float(threshold)
    if op == "le":
        return df[feature] <= float(threshold)
    return df[feature].astype(str) == str(threshold)


def search_rule_combinations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ranked: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked

    # Keep diverse top rules, avoiding many thresholds for the same feature/operator.
    candidates = (
        ranked.sort_values("expectancy_r", ascending=False)
        .drop_duplicates(subset=["feature", "operator"])
        .head(12)
        .to_dict("records")
    )

    rows: list[dict] = []
    rule_sets = [(r,) for r in candidates]
    rule_sets += list(combinations(candidates, 2))
    rule_sets += list(combinations(candidates[:8], 3))

    for rules in rule_sets:
        train_mask = pd.Series(True, index=train.index)
        test_mask = pd.Series(True, index=test.index)
        for rule in rules:
            train_mask &= _apply_rule(train, rule)
            test_mask &= _apply_rule(test, rule)

        train_subset = train[train_mask]
        test_subset = test[test_mask]
        if len(train_subset) < cfg.min_train_trades or len(test_subset) < cfg.min_test_trades:
            continue

        train_perf = performance_summary(train_subset)
        test_perf = performance_summary(test_subset)
        rows.append({
            "rules": " AND ".join(
                f"{r['feature']} {r['operator']} {r['threshold']}" for r in rules
            ),
            "rule_count": len(rules),
            **{f"train_{k}": v for k, v in train_perf.items()},
            **{f"test_{k}": v for k, v in test_perf.items()},
            "stability_score": min(train_perf["expectancy_r"], test_perf["expectancy_r"]),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["stability_score", "test_profit_factor", "test_trades"],
        ascending=[False, False, False],
    ).head(cfg.top_rules).reset_index(drop=True)

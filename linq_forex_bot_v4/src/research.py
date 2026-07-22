from __future__ import annotations

import pandas as pd


def target_probability_table(features: pd.DataFrame, ratios: list[float]) -> pd.DataFrame:
    rows = []
    n = len(features)
    for ratio in ratios:
        col = f"hit_{ratio:.2f}R"
        hits = int(features[col].sum()) if n and col in features else 0
        probability = hits / n if n else 0.0
        expectancy = probability * ratio - (1.0 - probability)
        rows.append({
            "ratio": ratio,
            "setups": n,
            "hits": hits,
            "hit_rate": probability,
            "theoretical_expectancy_r": expectancy,
        })
    return pd.DataFrame(rows)


def condition_breakdown(features: pd.DataFrame, selected_ratio: float) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["dimension", "value", "setups", "hit_rate", "average_mfe_r", "stop_rate"])
    target = f"hit_{selected_ratio:.2f}R"
    frames = []
    dimensions = {
        "direction": "direction",
        "weekday": "weekday",
        "hour_utc": "hour_utc",
        "trend_aligned": "trend_aligned",
        "bos_confirmed": "bos_confirmed",
    }
    for label, col in dimensions.items():
        grouped = features.groupby(col, dropna=False).agg(
            setups=(target, "size"),
            hit_rate=(target, "mean"),
            average_mfe_r=("max_favorable_r", "mean"),
            stop_rate=("stopped", "mean"),
        ).reset_index().rename(columns={col: "value"})
        grouped.insert(0, "dimension", label)
        frames.append(grouped)

    scored = features.copy()
    if scored["score"].nunique() > 1:
        scored["score_band"] = pd.qcut(scored["score"], q=min(4, scored["score"].nunique()), duplicates="drop")
        grouped = scored.groupby("score_band", observed=True).agg(
            setups=(target, "size"), hit_rate=(target, "mean"),
            average_mfe_r=("max_favorable_r", "mean"), stop_rate=("stopped", "mean"),
        ).reset_index().rename(columns={"score_band": "value"})
        grouped["value"] = grouped["value"].astype(str)
        grouped.insert(0, "dimension", "score_band")
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "feature"})

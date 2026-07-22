from __future__ import annotations

import numpy as np
import pandas as pd

from .research import performance_summary


def apply_rule_string(df: pd.DataFrame, rule_string: str) -> pd.Series:
    """Apply a rule written by the research engine to a setup dataframe."""
    mask = pd.Series(True, index=df.index)

    for clause in rule_string.split(" AND "):
        feature, operator, raw_value = clause.split(" ", 2)

        if feature not in df.columns:
            return pd.Series(False, index=df.index)

        if operator == "eq":
            mask &= df[feature].astype(str) == raw_value
        else:
            threshold = float(raw_value)

            if operator == "le":
                mask &= df[feature] <= threshold
            elif operator == "ge":
                mask &= df[feature] >= threshold
            else:
                raise ValueError(f"Unsupported operator: {operator}")

    return mask


def walk_forward_validate(
    labeled: pd.DataFrame,
    rule_string: str,
    folds: int = 4,
    min_fold_trades: int = 25,
) -> pd.DataFrame:
    """
    Evaluate one locked rule across consecutive unseen periods.

    Each fold tests a later period than the previous fold.
    """
    ordered = labeled.sort_values("timestamp").reset_index(drop=True)
    boundaries = np.linspace(0, len(ordered), folds + 2, dtype=int)

    rows: list[dict] = []

    for fold in range(folds):
        test_start = boundaries[fold + 1]
        test_end = boundaries[fold + 2]

        test_period = ordered.iloc[test_start:test_end]
        selected = test_period[apply_rule_string(test_period, rule_string)]
        performance = performance_summary(selected)

        rows.append(
            {
                "fold": fold + 1,
                "period_start": test_period["timestamp"].min(),
                "period_end": test_period["timestamp"].max(),
                "passes_min_trades": performance["trades"] >= min_fold_trades,
                **performance,
            }
        )

    return pd.DataFrame(rows)


def retrace_sensitivity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    center: float = 0.25,
    offsets: tuple[float, ...] = (-0.05, -0.025, 0.0, 0.025, 0.05),
) -> pd.DataFrame:
    """
    Test nearby retracement thresholds.

    A robust idea should remain useful near 0.25 instead of working only at
    exactly one threshold.
    """
    rows: list[dict] = []

    for offset in offsets:
        threshold = max(0.05, min(0.95, center + offset))

        train_subset = train[train["retrace_fraction"] <= threshold]
        test_subset = test[test["retrace_fraction"] <= threshold]

        train_performance = performance_summary(train_subset)
        test_performance = performance_summary(test_subset)

        rows.append(
            {
                "threshold": threshold,
                **{
                    f"train_{key}": value
                    for key, value in train_performance.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_performance.items()
                },
                "stability_score": min(
                    train_performance["expectancy_r"],
                    test_performance["expectancy_r"],
                ),
            }
        )

    return pd.DataFrame(rows)

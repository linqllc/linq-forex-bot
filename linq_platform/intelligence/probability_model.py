"""
Native probability model.

Migrated from the validated Phase 4 reference implementation.
"""

from __future__ import annotations

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import feature_columns


__all__ = [
    "make_model",
    "predict_holdout",
]


def make_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.25,
                    random_state=42,
                ),
            ),
        ]
    )


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
            continue

        model = make_model(X.iloc[:i])

        model.fit(
            X.iloc[:i],
            train_y,
        )

        probs[i] = model.predict_proba(X.iloc[i : i + 1])[0, 1]

    data["holdout"] = hold
    data["probability_1r"] = probs

    data["selected"] = (
        data["holdout"]
        & data["probability_1r"].notna()
        & (data["probability_1r"] >= cfg.probability_threshold)
    )

    return data

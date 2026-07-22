from __future__ import annotations

import html
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

try:
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

except ImportError as exc:
    raise SystemExit(
        "\nMissing a required package.\n\n"
        "Run this command first:\n\n"
        "python -m pip install pandas numpy scikit-learn joblib\n\n"
        f"Original error: {exc}\n"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIR = "EUR_USD"

REPORT_DIR = Path("reports/v8")
V7_REPORT_DIR = Path("reports/v7")

SETUPS_PATH = V7_REPORT_DIR / f"{PAIR}_automatic_setups.csv"
OUTCOMES_PATH = V7_REPORT_DIR / f"{PAIR}_outcomes.csv"

MODEL_PATH = REPORT_DIR / f"{PAIR}_prediction_engine.joblib"
PREDICTIONS_PATH = REPORT_DIR / f"{PAIR}_historical_predictions.csv"
FEATURE_PATH = REPORT_DIR / f"{PAIR}_feature_importance.csv"
SUMMARY_PATH = REPORT_DIR / f"{PAIR}_model_summary.json"
HTML_PATH = REPORT_DIR / f"{PAIR}_prediction_report.html"

RANDOM_STATE = 42
TEST_FRACTION = 0.30
MINIMUM_TRAIN_ROWS = 40

# A 3R trade has theoretical break-even probability of 25%.
# We use a higher research threshold to demand a margin of safety.
MINIMUM_3R_PROBABILITY = 0.35
MINIMUM_EXPECTED_VALUE_R = 0.20

LABELS = {
    "hit_1r": 1.0,
    "hit_2r": 2.0,
    "hit_3r": 3.0,
}

LEAKAGE_EXACT_COLUMNS = {
    "outcome",
    "realized_r",
    "mfe_r",
    "mae_r",
    "stop_hit",
    "bars_evaluated",
    "exit_price",
    "exit_timestamp",
    "exit_time",
    "close_reason",
    "profit",
    "loss",
    "pnl",
    "return_r",
    "target_hit",
}

LEAKAGE_TEXT_PATTERNS = (
    "outcome",
    "realized",
    "mfe",
    "mae",
    "stop_hit",
    "exit_",
    "future_",
    "forward_",
    "result",
    "profit",
    "pnl",
    "return_r",
    "bars_evaluated",
    "target_hit",
)

IDENTIFIER_PATTERNS = (
    "_id",
    "uuid",
    "identifier",
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def line(character: str = "=", length: int = 104) -> str:
    return character * length


def heading(title: str) -> None:
    print(f"\n{line()}")
    print(title)
    print(line())


def subheading(title: str) -> None:
    print(f"\n{title}")
    print(line("-", 104))


def format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def format_number(value: float | None, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def safe_metric(function, *args, **kwargs) -> float | None:
    try:
        return float(function(*args, **kwargs))
    except Exception:
        return None


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    return normalized.fillna(False).astype(bool)


# =============================================================================
# DATA LOADING
# =============================================================================

def require_files() -> None:
    missing = [
        path
        for path in (SETUPS_PATH, OUTCOMES_PATH)
        if not path.exists()
    ]

    if missing:
        message = "\nRequired files were not found:\n"

        for path in missing:
            message += f"\n  - {path}"

        message += (
            "\n\nRun the V7 detector and outcome engine first, "
            "then rerun this command.\n"
        )

        raise SystemExit(message)


def determine_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [
        "timestamp",
        "entry_timestamp",
        "entry_time",
        "setup_timestamp",
        "created_at",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise SystemExit(
        "No usable setup timestamp column was found.\n"
        "Expected one of: " + ", ".join(candidates)
    )


def load_historical_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    require_files()

    setups = pd.read_csv(SETUPS_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    if "setup_id" not in setups.columns:
        raise SystemExit(
            f"{SETUPS_PATH} does not contain a setup_id column."
        )

    if "setup_id" not in outcomes.columns:
        raise SystemExit(
            f"{OUTCOMES_PATH} does not contain a setup_id column."
        )

    missing_labels = [
        label
        for label in LABELS
        if label not in outcomes.columns
    ]

    if missing_labels:
        raise SystemExit(
            "Outcome file is missing required labels: "
            + ", ".join(missing_labels)
        )

    timestamp_column = determine_timestamp_column(setups)

    outcome_columns = ["setup_id", *LABELS.keys()]

    merged = setups.merge(
        outcomes[outcome_columns],
        on="setup_id",
        how="inner",
        validate="one_to_one",
    )

    merged[timestamp_column] = pd.to_datetime(
        merged[timestamp_column],
        utc=True,
        errors="coerce",
    )

    merged = (
        merged.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )

    for label in LABELS:
        merged[label] = normalize_bool(merged[label]).astype(int)

    if merged.empty:
        raise SystemExit(
            "No valid setup/outcome rows remained after merging."
        )

    return setups, merged, timestamp_column


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def is_leakage_column(column: str) -> bool:
    lowered = column.lower()

    if lowered in LEAKAGE_EXACT_COLUMNS:
        return True

    if lowered in LABELS:
        return True

    if any(pattern in lowered for pattern in LEAKAGE_TEXT_PATTERNS):
        return True

    return False


def is_identifier_column(column: str) -> bool:
    lowered = column.lower()

    if lowered == "setup_id":
        return True

    return any(
        lowered.endswith(pattern)
        for pattern in IDENTIFIER_PATTERNS
    )


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result[timestamp_column]

    result["entry_hour_utc"] = timestamp.dt.hour
    result["entry_weekday"] = timestamp.dt.day_name()
    result["entry_month"] = timestamp.dt.month
    result["entry_day_of_month"] = timestamp.dt.day
    result["is_london_session"] = (
        timestamp.dt.hour.between(7, 11)
    ).astype(int)
    result["is_new_york_session"] = (
        timestamp.dt.hour.between(12, 16)
    ).astype(int)
    result["is_london_new_york_overlap"] = (
        timestamp.dt.hour.between(12, 15)
    ).astype(int)

    return result


def add_zone_lifecycle_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "touches" in result.columns:
        result["touches"] = pd.to_numeric(
            result["touches"],
            errors="coerce",
        )

        result["is_first_touch"] = (
            result["touches"] <= 1
        ).astype(int)

        result["is_second_touch"] = (
            result["touches"] == 2
        ).astype(int)

        result["is_mature_zone"] = (
            result["touches"] >= 3
        ).astype(int)

    if "fresh_zone" in result.columns:
        result["fresh_zone"] = normalize_bool(
            result["fresh_zone"]
        ).astype(int)

    return result


def convert_boolean_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in result.columns:
        series = result[column]

        if pd.api.types.is_bool_dtype(series):
            result[column] = series.astype(int)
            continue

        if series.dtype != object:
            continue

        values = {
            str(value).strip().lower()
            for value in series.dropna().unique()
        }

        boolean_values = {
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "y",
            "n",
        }

        if values and values.issubset(boolean_values):
            result[column] = normalize_bool(series).astype(int)

    return result


def prepare_features(
    merged: pd.DataFrame,
    timestamp_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    working = add_time_features(merged, timestamp_column)
    working = add_zone_lifecycle_features(working)
    working = convert_boolean_like_columns(working)

    excluded_columns: list[str] = []

    for column in working.columns:
        if column == timestamp_column:
            excluded_columns.append(column)
            continue

        if is_leakage_column(column):
            excluded_columns.append(column)
            continue

        if is_identifier_column(column):
            excluded_columns.append(column)

    feature_columns = [
        column
        for column in working.columns
        if column not in excluded_columns
    ]

    features = working[feature_columns].copy()

    # Drop columns with no variation because they cannot help prediction.
    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(dropna=False) <= 1
    ]

    features = features.drop(columns=constant_columns)
    excluded_columns.extend(constant_columns)

    # Drop columns where every row is unique text, which are usually IDs,
    # notes, raw timestamps, or descriptions rather than stable features.
    high_cardinality_text = []

    for column in features.select_dtypes(include=["object"]).columns:
        unique_ratio = features[column].nunique(dropna=True) / max(
            len(features),
            1,
        )

        if unique_ratio > 0.80:
            high_cardinality_text.append(column)

    features = features.drop(columns=high_cardinality_text)
    excluded_columns.extend(high_cardinality_text)

    if features.empty:
        raise SystemExit(
            "No usable pre-entry features were found after leakage removal."
        )

    return features, sorted(set(excluded_columns))


# =============================================================================
# MODEL CREATION
# =============================================================================

def build_preprocessor(
    features: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def candidate_models(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# =============================================================================
# TIME-SERIES VALIDATION
# =============================================================================

def valid_time_series_splits(
    y: pd.Series,
    requested_splits: int = 4,
) -> int:
    maximum = min(requested_splits, len(y) - 1)

    for splits in range(maximum, 1, -1):
        splitter = TimeSeriesSplit(n_splits=splits)
        valid = True

        for train_index, validation_index in splitter.split(y):
            train_labels = y.iloc[train_index]
            validation_labels = y.iloc[validation_index]

            if train_labels.nunique() < 2:
                valid = False
                break

            if validation_labels.nunique() < 2:
                valid = False
                break

        if valid:
            return splits

    return 0


def cross_validate_model(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, Any]:
    splits = valid_time_series_splits(labels)

    if splits < 2:
        return {
            "folds": 0,
            "brier_score": None,
            "log_loss": None,
            "roc_auc": None,
        }

    splitter = TimeSeriesSplit(n_splits=splits)

    brier_scores = []
    log_losses = []
    auc_scores = []

    for train_index, validation_index in splitter.split(features):
        x_train = features.iloc[train_index]
        x_validation = features.iloc[validation_index]

        y_train = labels.iloc[train_index]
        y_validation = labels.iloc[validation_index]

        fitted = clone(model)
        fitted.fit(x_train, y_train)

        probabilities = fitted.predict_proba(x_validation)[:, 1]

        brier_scores.append(
            brier_score_loss(y_validation, probabilities)
        )

        log_losses.append(
            log_loss(
                y_validation,
                probabilities,
                labels=[0, 1],
            )
        )

        auc = safe_metric(
            roc_auc_score,
            y_validation,
            probabilities,
        )

        if auc is not None:
            auc_scores.append(auc)

    return {
        "folds": splits,
        "brier_score": float(np.mean(brier_scores)),
        "log_loss": float(np.mean(log_losses)),
        "roc_auc": (
            float(np.mean(auc_scores))
            if auc_scores
            else None
        ),
    }


def choose_model(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[str, Pipeline, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    models = candidate_models(preprocessor)

    for name, model in models.items():
        results[name] = cross_validate_model(
            model,
            features,
            labels,
        )

    ranked = sorted(
        models.keys(),
        key=lambda name: (
            results[name]["brier_score"]
            if results[name]["brier_score"] is not None
            else float("inf")
        ),
    )

    selected_name = ranked[0]

    return selected_name, models[selected_name], results


# =============================================================================
# HOLDOUT EVALUATION
# =============================================================================

def chronological_split(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    split_index = int(len(features) * (1 - TEST_FRACTION))
    split_index = max(split_index, MINIMUM_TRAIN_ROWS)
    split_index = min(split_index, len(features) - 10)

    if split_index <= 0 or split_index >= len(features):
        raise SystemExit(
            "Not enough rows to create a chronological train/test split."
        )

    x_train = features.iloc[:split_index].copy()
    x_test = features.iloc[split_index:].copy()

    y_train = labels.iloc[:split_index].copy()
    y_test = labels.iloc[split_index:].copy()

    return x_train, x_test, y_train, y_test, split_index


def evaluate_probabilities(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)

    return {
        "rows": int(len(labels)),
        "positive_examples": int(labels.sum()),
        "base_rate": float(labels.mean()),
        "average_predicted_probability": float(
            np.mean(probabilities)
        ),
        "brier_score": safe_metric(
            brier_score_loss,
            labels,
            probabilities,
        ),
        "log_loss": safe_metric(
            log_loss,
            labels,
            probabilities,
            labels=[0, 1],
        ),
        "roc_auc": safe_metric(
            roc_auc_score,
            labels,
            probabilities,
        ),
        "accuracy": safe_metric(
            accuracy_score,
            labels,
            predictions,
        ),
        "precision": safe_metric(
            precision_score,
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": safe_metric(
            recall_score,
            labels,
            predictions,
            zero_division=0,
        ),
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    if len(features) < 10 or labels.nunique() < 2:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_std",
            ]
        )

    try:
        result = permutation_importance(
            model,
            features,
            labels,
            scoring="neg_brier_score",
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importance = pd.DataFrame(
            {
                "feature": features.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

        return importance.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    except Exception:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_std",
            ]
        )


# =============================================================================
# SIGNAL CREATION
# =============================================================================

def expected_value(probability: pd.Series, target_r: float) -> pd.Series:
    return probability * target_r - (1 - probability)


def confidence_label(probability: float) -> str:
    if probability >= 0.55:
        return "HIGH"
    if probability >= 0.40:
        return "MODERATE"
    if probability >= 0.30:
        return "WATCH"
    return "LOW"


def recommendation_label(
    probability_3r: float,
    expected_value_3r: float,
) -> str:
    if (
        probability_3r >= MINIMUM_3R_PROBABILITY
        and expected_value_3r >= MINIMUM_EXPECTED_VALUE_R
    ):
        return "RESEARCH CANDIDATE"

    return "PASS"


# =============================================================================
# REPORT BUILDING
# =============================================================================

def metric_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("Rows", metrics.get("rows")),
        ("Actual success rate", format_pct(metrics.get("base_rate"))),
        (
            "Average prediction",
            format_pct(metrics.get("average_predicted_probability")),
        ),
        ("Brier score", format_number(metrics.get("brier_score"))),
        ("Log loss", format_number(metrics.get("log_loss"))),
        ("ROC AUC", format_number(metrics.get("roc_auc"))),
        ("Accuracy", format_pct(metrics.get("accuracy"))),
        ("Precision", format_pct(metrics.get("precision"))),
        ("Recall", format_pct(metrics.get("recall"))),
    ]

    body = "".join(
        f"<tr><td>{html.escape(str(name))}</td>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for name, value in rows
    )

    return (
        "<table class='metrics'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def build_html_report(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    timestamp_column: str,
) -> str:
    latest_columns = [
        timestamp_column,
        "direction",
        "grade",
        "confluence_score",
        "probability_1r",
        "probability_2r",
        "probability_3r",
        "expected_value_3r",
        "confidence",
        "recommendation",
    ]

    available_latest_columns = [
        column
        for column in latest_columns
        if column in predictions.columns
    ]

    latest = (
        predictions[available_latest_columns]
        .tail(25)
        .sort_values(
            "probability_3r",
            ascending=False,
        )
    )

    candidates = predictions[
        predictions["recommendation"] == "RESEARCH CANDIDATE"
    ].copy()

    candidates = candidates.sort_values(
        [
            "probability_3r",
            "expected_value_3r",
        ],
        ascending=False,
    ).head(25)

    candidate_columns = [
        column
        for column in latest_columns
        if column in candidates.columns
    ]

    importance_html = (
        importance.head(20).to_html(
            index=False,
            classes="data-table",
            border=0,
            float_format=lambda value: f"{value:.4f}",
        )
        if not importance.empty
        else "<p>No stable feature-importance result was available.</p>"
    )

    candidate_html = (
        candidates[candidate_columns].to_html(
            index=False,
            classes="data-table",
            border=0,
            float_format=lambda value: f"{value:.3f}",
        )
        if not candidates.empty
        else (
            "<div class='notice'>"
            "No historical rows passed the current research threshold."
            "</div>"
        )
    )

    latest_html = latest.to_html(
        index=False,
        classes="data-table",
        border=0,
        float_format=lambda value: f"{value:.3f}",
    )

    holdout_metrics = summary["targets"]["hit_3r"]["holdout_metrics"]

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LINQ V8 Prediction Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        background: #f4f6f8;
        color: #18212b;
    }}

    .container {{
        max-width: 1320px;
        margin: 0 auto;
        padding: 32px 20px 60px;
    }}

    h1 {{
        margin-bottom: 4px;
    }}

    h2 {{
        margin-top: 0;
    }}

    .subtitle {{
        color: #5b6572;
        margin-bottom: 28px;
    }}

    .cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 14px;
        margin-bottom: 28px;
    }}

    .card {{
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }}

    .card-label {{
        font-size: 13px;
        color: #657180;
        margin-bottom: 8px;
    }}

    .card-value {{
        font-size: 26px;
        font-weight: 700;
    }}

    .section {{
        background: white;
        border-radius: 12px;
        margin-top: 20px;
        padding: 22px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        overflow-x: auto;
    }}

    .notice {{
        padding: 16px;
        background: #fff6d8;
        border-left: 4px solid #d9a400;
        border-radius: 6px;
    }}

    .warning {{
        padding: 16px;
        background: #fdecec;
        border-left: 4px solid #c43b3b;
        border-radius: 6px;
        margin-bottom: 24px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    th {{
        text-align: left;
        background: #eef1f4;
        padding: 10px;
        white-space: nowrap;
    }}

    td {{
        border-bottom: 1px solid #e9edf1;
        padding: 10px;
        white-space: nowrap;
    }}

    .metrics {{
        max-width: 620px;
    }}

    code {{
        background: #eef1f4;
        padding: 2px 5px;
        border-radius: 4px;
    }}
</style>
</head>

<body>
<div class="container">
    <h1>LINQ Forex Bot V8</h1>
    <div class="subtitle">
        Historical-learning reversal prediction report — {PAIR}
    </div>

    <div class="warning">
        <strong>Research system only.</strong>
        This report uses a small historical sample and must not be treated as
        live-trading validation. Out-of-sample and walk-forward testing are
        still required.
    </div>

    <div class="cards">
        <div class="card">
            <div class="card-label">Historical examples</div>
            <div class="card-value">{summary["rows"]}</div>
        </div>

        <div class="card">
            <div class="card-label">Training examples</div>
            <div class="card-value">{summary["training_rows"]}</div>
        </div>

        <div class="card">
            <div class="card-label">Holdout examples</div>
            <div class="card-value">{summary["holdout_rows"]}</div>
        </div>

        <div class="card">
            <div class="card-label">Usable features</div>
            <div class="card-value">{summary["feature_count"]}</div>
        </div>

        <div class="card">
            <div class="card-label">Selected 3R model</div>
            <div class="card-value" style="font-size:18px;">
                {html.escape(summary["targets"]["hit_3r"]["selected_model"])}
            </div>
        </div>

        <div class="card">
            <div class="card-label">Holdout 3R AUC</div>
            <div class="card-value">
                {format_number(holdout_metrics.get("roc_auc"))}
            </div>
        </div>
    </div>

    <div class="section">
        <h2>What V8 is estimating</h2>
        <p>
            For each historical setup, V8 estimates the probability that price
            reaches <strong>1R, 2R, and 3R before the stop</strong>.
        </p>
        <p>
            Freshness is retained as descriptive data, but it receives no
            automatic point bonus. The models are allowed to learn whether
            first-touch, second-touch, or mature zones performed better in the
            available history.
        </p>
    </div>

    <div class="section">
        <h2>3R holdout performance</h2>
        {metric_table(holdout_metrics)}
    </div>

    <div class="section">
        <h2>Historical research candidates</h2>
        <p>
            These rows crossed the current probability and expected-value
            thresholds. They are examples for research, not live trade calls.
        </p>
        {candidate_html}
    </div>

    <div class="section">
        <h2>Most influential 3R features</h2>
        {importance_html}
    </div>

    <div class="section">
        <h2>Latest 25 historical setups</h2>
        {latest_html}
    </div>

    <div class="section">
        <h2>Files generated</h2>
        <p><code>{MODEL_PATH}</code></p>
        <p><code>{PREDICTIONS_PATH}</code></p>
        <p><code>{FEATURE_PATH}</code></p>
        <p><code>{SUMMARY_PATH}</code></p>
    </div>
</div>
</body>
</html>
"""


# =============================================================================
# MAIN ENGINE
# =============================================================================

def main() -> None:
    heading("LINQ FOREX BOT V8 — HISTORICAL LEARNING ENGINE")

    _, merged, timestamp_column = load_historical_data()
    features, excluded_columns = prepare_features(
        merged,
        timestamp_column,
    )

    if len(features) < MINIMUM_TRAIN_ROWS + 10:
        raise SystemExit(
            f"Only {len(features)} examples are available. "
            f"At least {MINIMUM_TRAIN_ROWS + 10} are required."
        )

    preprocessor, numeric_columns, categorical_columns = (
        build_preprocessor(features)
    )

    print(f"Pair:                         {PAIR}")
    print(f"Historical examples:          {len(merged)}")
    print(f"Usable pre-entry features:    {features.shape[1]}")
    print(f"Numeric features:             {len(numeric_columns)}")
    print(f"Categorical features:         {len(categorical_columns)}")
    print(f"Excluded/leakage columns:     {len(excluded_columns)}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = merged.copy()
    model_bundle: dict[str, Any] = {
        "pair": PAIR,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": features.columns.tolist(),
        "excluded_columns": excluded_columns,
        "timestamp_column": timestamp_column,
        "models": {},
    }

    target_summaries: dict[str, Any] = {}
    split_index_used = None
    three_r_importance = pd.DataFrame()

    for label, target_r in LABELS.items():
        heading(f"TRAINING TARGET: {label.upper()}")

        labels = merged[label].astype(int)

        (
            x_train,
            x_test,
            y_train,
            y_test,
            split_index,
        ) = chronological_split(features, labels)

        split_index_used = split_index

        print(f"Training rows:                {len(x_train)}")
        print(f"Chronological holdout rows:   {len(x_test)}")
        print(
            f"Training success rate:        "
            f"{format_pct(y_train.mean())}"
        )
        print(
            f"Holdout success rate:         "
            f"{format_pct(y_test.mean())}"
        )

        selected_name, selected_model, cv_results = choose_model(
            preprocessor,
            x_train,
            y_train,
        )

        subheading("Time-series model comparison")

        for model_name, result in cv_results.items():
            print(
                f"{model_name:<24}"
                f" Brier={format_number(result['brier_score'])}"
                f"  LogLoss={format_number(result['log_loss'])}"
                f"  AUC={format_number(result['roc_auc'])}"
                f"  folds={result['folds']}"
            )

        print(f"\nSelected model: {selected_name}")

        selected_model.fit(x_train, y_train)

        holdout_probabilities = selected_model.predict_proba(
            x_test
        )[:, 1]

        threshold = (
            MINIMUM_3R_PROBABILITY
            if label == "hit_3r"
            else 0.50
        )

        holdout_metrics = evaluate_probabilities(
            y_test,
            holdout_probabilities,
            threshold,
        )

        subheading("Chronological holdout performance")

        print(
            f"Actual success rate:          "
            f"{format_pct(holdout_metrics['base_rate'])}"
        )
        print(
            f"Average predicted probability:"
            f" {format_pct(holdout_metrics['average_predicted_probability'])}"
        )
        print(
            f"Brier score:                  "
            f"{format_number(holdout_metrics['brier_score'])}"
        )
        print(
            f"ROC AUC:                      "
            f"{format_number(holdout_metrics['roc_auc'])}"
        )
        print(
            f"Precision:                    "
            f"{format_pct(holdout_metrics['precision'])}"
        )
        print(
            f"Recall:                       "
            f"{format_pct(holdout_metrics['recall'])}"
        )

        # Refit on all historical examples after holdout evaluation.
        final_model = clone(selected_model)
        final_model.fit(features, labels)

        all_probabilities = final_model.predict_proba(
            features
        )[:, 1]

        probability_column = f"probability_{int(target_r)}r"
        predictions[probability_column] = all_probabilities

        model_bundle["models"][label] = final_model

        target_summaries[label] = {
            "target_r": target_r,
            "selected_model": selected_name,
            "historical_success_rate": float(labels.mean()),
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
        }

        if label == "hit_3r":
            three_r_importance = calculate_feature_importance(
                selected_model,
                x_test,
                y_test,
            )

    predictions["expected_value_1r"] = expected_value(
        predictions["probability_1r"],
        1.0,
    )

    predictions["expected_value_2r"] = expected_value(
        predictions["probability_2r"],
        2.0,
    )

    predictions["expected_value_3r"] = expected_value(
        predictions["probability_3r"],
        3.0,
    )

    predictions["confidence"] = predictions[
        "probability_3r"
    ].map(confidence_label)

    predictions["recommendation"] = predictions.apply(
        lambda row: recommendation_label(
            row["probability_3r"],
            row["expected_value_3r"],
        ),
        axis=1,
    )

    predictions["data_partition"] = "training"
    predictions.loc[
        predictions.index >= split_index_used,
        "data_partition",
    ] = "chronological_holdout"

    summary = {
        "pair": PAIR,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(merged)),
        "training_rows": int(split_index_used),
        "holdout_rows": int(len(merged) - split_index_used),
        "feature_count": int(features.shape[1]),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "features": features.columns.tolist(),
        "excluded_columns": excluded_columns,
        "research_thresholds": {
            "minimum_3r_probability": MINIMUM_3R_PROBABILITY,
            "minimum_expected_value_3r": MINIMUM_EXPECTED_VALUE_R,
        },
        "targets": target_summaries,
    }

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    three_r_importance.to_csv(
        FEATURE_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2)
    )

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    report = build_html_report(
        summary,
        predictions,
        three_r_importance,
        timestamp_column,
    )

    HTML_PATH.write_text(report)

    heading("V8 RESEARCH SIGNAL SUMMARY")

    research_candidates = predictions[
        predictions["recommendation"] == "RESEARCH CANDIDATE"
    ].copy()

    print(
        f"Historical research candidates: "
        f"{len(research_candidates)} / {len(predictions)}"
    )

    if research_candidates.empty:
        print(
            "\nNo rows crossed both the probability and "
            "expected-value thresholds."
        )
    else:
        display_columns = [
            timestamp_column,
            "direction",
            "grade",
            "confluence_score",
            "probability_1r",
            "probability_2r",
            "probability_3r",
            "expected_value_3r",
            "confidence",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in research_candidates.columns
        ]

        top_candidates = research_candidates.sort_values(
            [
                "probability_3r",
                "expected_value_3r",
            ],
            ascending=False,
        ).head(15)

        print(
            "\n"
            + top_candidates[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    heading("FILES SAVED")

    print(f"Readable HTML report:          {HTML_PATH}")
    print(f"Historical predictions:        {PREDICTIONS_PATH}")
    print(f"Feature importance:            {FEATURE_PATH}")
    print(f"Model summary:                 {SUMMARY_PATH}")
    print(f"Saved prediction models:       {MODEL_PATH}")

    print(
        "\nOpen the report on your Mac with:\n\n"
        f"open {HTML_PATH}\n"
    )

    print(
        line()
        + "\nV8 completed successfully.\n"
        + line()
    )


if __name__ == "__main__":
    main()

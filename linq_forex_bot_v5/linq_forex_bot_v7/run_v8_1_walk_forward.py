from __future__ import annotations

import html
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "\nMissing packages.\n\n"
        "Run:\n"
        "python -m pip install pandas numpy scikit-learn\n\n"
        f"Original error: {exc}"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIR = "EUR_USD"

V7_DIR = Path("reports/v7")
REPORT_DIR = Path("reports/v8_1")

SETUPS_PATH = V7_DIR / f"{PAIR}_automatic_setups.csv"
OUTCOMES_PATH = V7_DIR / f"{PAIR}_outcomes.csv"

PREDICTIONS_PATH = REPORT_DIR / f"{PAIR}_walk_forward_predictions.csv"
TRADES_PATH = REPORT_DIR / f"{PAIR}_walk_forward_trades.csv"
SUMMARY_PATH = REPORT_DIR / f"{PAIR}_walk_forward_summary.json"
HTML_PATH = REPORT_DIR / f"{PAIR}_walk_forward_report.html"

MINIMUM_TRAIN_ROWS = 40

# Research thresholds. These are fixed before prediction and are not optimized
# on future results.
MINIMUM_1R_PROBABILITY = 0.55
MINIMUM_3R_PROBABILITY = 0.35
MINIMUM_EXPECTED_VALUE_3R = 0.20

TARGETS = {
    "hit_1r": 1,
    "hit_2r": 2,
    "hit_3r": 3,
}

LEAKAGE_COLUMNS = {
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

LEAKAGE_PATTERNS = (
    "outcome",
    "realized",
    "mfe",
    "mae",
    "stop_hit",
    "exit_",
    "future_",
    "forward_",
    "profit",
    "pnl",
    "return_r",
    "bars_evaluated",
    "target_hit",
)


# =============================================================================
# HELPERS
# =============================================================================

def divider(character: str = "=", length: int = 108) -> str:
    return character * length


def heading(title: str) -> None:
    print(f"\n{divider()}")
    print(title)
    print(divider())


def subheading(title: str) -> None:
    print(f"\n{title}")
    print(divider("-", 108))


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


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


def find_timestamp_column(frame: pd.DataFrame) -> str:
    candidates = [
        "timestamp",
        "entry_timestamp",
        "entry_time",
        "setup_timestamp",
        "created_at",
    ]

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise SystemExit(
        "No timestamp column found. Expected one of: "
        + ", ".join(candidates)
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> tuple[pd.DataFrame, str]:
    missing = [
        path
        for path in (SETUPS_PATH, OUTCOMES_PATH)
        if not path.exists()
    ]

    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "\nRequired files are missing:\n"
            f"{lines}\n\n"
            "Run the V7 setup detector and outcome engine first."
        )

    setups = pd.read_csv(SETUPS_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    for name, frame in (("setups", setups), ("outcomes", outcomes)):
        if "setup_id" not in frame.columns:
            raise SystemExit(
                f"The {name} file does not contain setup_id."
            )

    missing_targets = [
        target
        for target in TARGETS
        if target not in outcomes.columns
    ]

    if missing_targets:
        raise SystemExit(
            "Outcome file is missing: "
            + ", ".join(missing_targets)
        )

    timestamp_column = find_timestamp_column(setups)

    merged = setups.merge(
        outcomes[["setup_id", *TARGETS.keys()]],
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
        .drop_duplicates("setup_id")
        .reset_index(drop=True)
    )

    for target in TARGETS:
        merged[target] = normalize_boolean(merged[target])

    if len(merged) <= MINIMUM_TRAIN_ROWS:
        raise SystemExit(
            f"Only {len(merged)} examples were loaded. "
            f"More than {MINIMUM_TRAIN_ROWS} are required."
        )

    return merged, timestamp_column


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def is_leakage(column: str) -> bool:
    lowered = column.lower()

    if lowered in LEAKAGE_COLUMNS:
        return True

    if lowered in TARGETS:
        return True

    return any(pattern in lowered for pattern in LEAKAGE_PATTERNS)


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result[timestamp_column]

    result["entry_hour_utc"] = timestamp.dt.hour
    result["entry_weekday"] = timestamp.dt.day_name()
    result["entry_month"] = timestamp.dt.month

    result["london_session"] = (
        timestamp.dt.hour.between(7, 11)
    ).astype(int)

    result["new_york_session"] = (
        timestamp.dt.hour.between(12, 16)
    ).astype(int)

    result["london_new_york_overlap"] = (
        timestamp.dt.hour.between(12, 15)
    ).astype(int)

    return result


def add_zone_lifecycle_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "touches" in result.columns:
        touches = pd.to_numeric(
            result["touches"],
            errors="coerce",
        )

        result["touches"] = touches
        result["is_first_touch"] = (touches <= 1).astype(int)
        result["is_second_touch"] = (touches == 2).astype(int)
        result["is_third_or_later_touch"] = (touches >= 3).astype(int)

    if "fresh_zone" in result.columns:
        result["fresh_zone"] = normalize_boolean(
            result["fresh_zone"]
        )

    return result


def convert_boolean_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    allowed = {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "1",
        "0",
    }

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

        if values and values.issubset(allowed):
            result[column] = normalize_boolean(series)

    return result


def prepare_features(
    merged: pd.DataFrame,
    timestamp_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    working = add_time_features(merged, timestamp_column)
    working = add_zone_lifecycle_features(working)
    working = convert_boolean_like_columns(working)

    excluded: list[str] = []

    for column in working.columns:
        lowered = column.lower()

        if column == timestamp_column:
            excluded.append(column)
        elif column == "setup_id":
            excluded.append(column)
        elif lowered.endswith("_id"):
            excluded.append(column)
        elif "uuid" in lowered:
            excluded.append(column)
        elif is_leakage(column):
            excluded.append(column)

    features = working.drop(
        columns=list(dict.fromkeys(excluded)),
        errors="ignore",
    )

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(dropna=False) <= 1
    ]

    features = features.drop(columns=constant_columns)
    excluded.extend(constant_columns)

    high_cardinality_text = []

    for column in features.select_dtypes(include=["object"]).columns:
        unique_ratio = (
            features[column].nunique(dropna=True)
            / max(len(features), 1)
        )

        if unique_ratio > 0.80:
            high_cardinality_text.append(column)

    features = features.drop(columns=high_cardinality_text)
    excluded.extend(high_cardinality_text)

    if features.empty:
        raise SystemExit(
            "No usable pre-entry features remained."
        )

    return features, sorted(set(excluded))


# =============================================================================
# MODEL
# =============================================================================

def create_model(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
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
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Logistic regression is intentionally used instead of the random forest.
    # The current dataset is very small and the random forest produced highly
    # overconfident in-sample probabilities.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=0.25,
                    random_state=42,
                ),
            ),
        ]
    )


# =============================================================================
# TRUE WALK-FORWARD PREDICTION
# =============================================================================

def walk_forward_predict(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[np.ndarray, list[int]]:
    probabilities = np.full(len(features), np.nan)
    training_sizes = [0] * len(features)

    for prediction_index in range(
        MINIMUM_TRAIN_ROWS,
        len(features),
    ):
        x_train = features.iloc[:prediction_index]
        y_train = labels.iloc[:prediction_index]
        x_predict = features.iloc[
            prediction_index:prediction_index + 1
        ]

        training_sizes[prediction_index] = prediction_index

        # A classifier cannot be trained if all earlier outcomes are identical.
        if y_train.nunique() < 2:
            probabilities[prediction_index] = float(
                y_train.mean()
            )
            continue

        model = create_model(x_train)
        model.fit(x_train, y_train)

        probabilities[prediction_index] = model.predict_proba(
            x_predict
        )[0, 1]

    return probabilities, training_sizes


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    valid = probabilities.notna()

    actual = labels.loc[valid].astype(int)
    predicted_probability = probabilities.loc[valid].astype(float)

    if len(actual) == 0:
        return {}

    predicted_class = (
        predicted_probability >= threshold
    ).astype(int)

    return {
        "rows": int(len(actual)),
        "wins": int(actual.sum()),
        "success_rate": float(actual.mean()),
        "average_probability": float(
            predicted_probability.mean()
        ),
        "brier_score": safe_metric(
            brier_score_loss,
            actual,
            predicted_probability,
        ),
        "log_loss": safe_metric(
            log_loss,
            actual,
            predicted_probability,
            labels=[0, 1],
        ),
        "roc_auc": safe_metric(
            roc_auc_score,
            actual,
            predicted_probability,
        ),
        "accuracy": safe_metric(
            accuracy_score,
            actual,
            predicted_class,
        ),
        "precision": safe_metric(
            precision_score,
            actual,
            predicted_class,
            zero_division=0,
        ),
        "recall": safe_metric(
            recall_score,
            actual,
            predicted_class,
            zero_division=0,
        ),
        "predicted_positive_count": int(
            predicted_class.sum()
        ),
    }


def expected_value_3r(probability: pd.Series) -> pd.Series:
    return probability * 3 - (1 - probability)


def confidence_label(probability: float) -> str:
    if pd.isna(probability):
        return "NOT PREDICTED"
    if probability >= 0.55:
        return "HIGH"
    if probability >= 0.40:
        return "MODERATE"
    if probability >= 0.30:
        return "WATCH"
    return "LOW"


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def build_trade_log(
    predictions: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    trade_mask = (
        predictions["probability_1r"].notna()
        & predictions["probability_3r"].notna()
        & (
            predictions["probability_1r"]
            >= MINIMUM_1R_PROBABILITY
        )
        & (
            predictions["probability_3r"]
            >= MINIMUM_3R_PROBABILITY
        )
        & (
            predictions["expected_value_3r"]
            >= MINIMUM_EXPECTED_VALUE_3R
        )
    )

    trades = predictions.loc[trade_mask].copy()

    trades["realized_r"] = np.where(
        trades["hit_3r"].astype(bool),
        3.0,
        -1.0,
    )

    trades["cumulative_r"] = trades["realized_r"].cumsum()
    trades["equity_peak_r"] = trades["cumulative_r"].cummax()
    trades["drawdown_r"] = (
        trades["cumulative_r"] - trades["equity_peak_r"]
    )

    trades["week"] = (
        trades[timestamp_column]
        .dt.to_period("W")
        .astype(str)
    )

    return trades


def summarize_trades(
    trades: pd.DataFrame,
    predictions: pd.DataFrame,
    timestamp_column: str,
) -> dict[str, Any]:
    eligible = predictions[
        predictions["probability_3r"].notna()
    ].copy()

    if eligible.empty:
        return {}

    first_date = eligible[timestamp_column].min()
    last_date = eligible[timestamp_column].max()

    total_days = max(
        (last_date - first_date).total_seconds() / 86400,
        1,
    )

    total_weeks = max(total_days / 7, 1)

    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "maximum_drawdown_r": 0.0,
            "trades_per_week": 0.0,
            "weeks_with_trade": 0,
            "eligible_weeks": int(np.ceil(total_weeks)),
            "percent_weeks_with_trade": 0.0,
        }

    wins = int((trades["realized_r"] > 0).sum())
    losses = int((trades["realized_r"] < 0).sum())

    gross_profit = float(
        trades.loc[
            trades["realized_r"] > 0,
            "realized_r",
        ].sum()
    )

    gross_loss = abs(
        float(
            trades.loc[
                trades["realized_r"] < 0,
                "realized_r",
            ].sum()
        )
    )

    eligible_weeks = int(
        eligible[timestamp_column]
        .dt.to_period("W")
        .nunique()
    )

    weeks_with_trade = int(
        trades[timestamp_column]
        .dt.to_period("W")
        .nunique()
    )

    return {
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(trades)),
        "total_r": float(trades["realized_r"].sum()),
        "expectancy_r": float(trades["realized_r"].mean()),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        ),
        "maximum_drawdown_r": abs(
            float(trades["drawdown_r"].min())
        ),
        "trades_per_week": float(len(trades) / total_weeks),
        "weeks_with_trade": weeks_with_trade,
        "eligible_weeks": eligible_weeks,
        "percent_weeks_with_trade": (
            weeks_with_trade / eligible_weeks
            if eligible_weeks > 0
            else 0.0
        ),
    }


# =============================================================================
# CALIBRATION TABLE
# =============================================================================

def calibration_table(
    labels: pd.Series,
    probabilities: pd.Series,
) -> pd.DataFrame:
    valid = probabilities.notna()

    frame = pd.DataFrame(
        {
            "actual": labels.loc[valid].astype(int),
            "probability": probabilities.loc[valid].astype(float),
        }
    )

    if frame.empty:
        return pd.DataFrame()

    bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 1.01]
    names = [
        "0–20%",
        "20–30%",
        "30–40%",
        "40–50%",
        "50–60%",
        "60%+",
    ]

    frame["probability_band"] = pd.cut(
        frame["probability"],
        bins=bins,
        labels=names,
        include_lowest=True,
        right=False,
    )

    grouped = (
        frame.groupby(
            "probability_band",
            observed=False,
        )
        .agg(
            predictions=("actual", "size"),
            winners=("actual", "sum"),
            actual_win_rate=("actual", "mean"),
            average_prediction=("probability", "mean"),
        )
        .reset_index()
    )

    return grouped


# =============================================================================
# REPORT
# =============================================================================

def metric_cards(
    trade_summary: dict[str, Any],
) -> str:
    cards = [
        ("Trades taken", trade_summary.get("trades", 0)),
        ("Wins", trade_summary.get("wins", 0)),
        (
            "Win rate",
            format_pct(trade_summary.get("win_rate")),
        ),
        (
            "Total return",
            f"{format_number(trade_summary.get('total_r'))}R",
        ),
        (
            "Expectancy",
            f"{format_number(trade_summary.get('expectancy_r'))}R",
        ),
        (
            "Profit factor",
            format_number(trade_summary.get("profit_factor")),
        ),
        (
            "Maximum drawdown",
            f"{format_number(trade_summary.get('maximum_drawdown_r'))}R",
        ),
        (
            "Trades per week",
            format_number(
                trade_summary.get("trades_per_week"),
                2,
            ),
        ),
        (
            "Weeks with a trade",
            format_pct(
                trade_summary.get(
                    "percent_weeks_with_trade"
                )
            ),
        ),
    ]

    return "".join(
        f"""
        <div class="card">
            <div class="label">{html.escape(str(label))}</div>
            <div class="value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in cards
    )


def metrics_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("Predictions", metrics.get("rows")),
        ("Actual winners", metrics.get("wins")),
        (
            "Actual success rate",
            format_pct(metrics.get("success_rate")),
        ),
        (
            "Average predicted probability",
            format_pct(metrics.get("average_probability")),
        ),
        (
            "Brier score",
            format_number(metrics.get("brier_score")),
        ),
        (
            "ROC AUC",
            format_number(metrics.get("roc_auc")),
        ),
        (
            "Precision",
            format_pct(metrics.get("precision")),
        ),
        (
            "Recall",
            format_pct(metrics.get("recall")),
        ),
    ]

    body = "".join(
        f"<tr><td>{html.escape(str(name))}</td>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for name, value in rows
    )

    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def build_html(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    target_metrics: dict[str, Any],
    trade_summary: dict[str, Any],
    calibration: pd.DataFrame,
    timestamp_column: str,
) -> str:
    trade_columns = [
        timestamp_column,
        "direction",
        "grade",
        "confluence_score",
        "touches",
        "fresh_zone",
        "probability_1r",
        "probability_2r",
        "probability_3r",
        "expected_value_3r",
        "hit_3r",
        "realized_r",
        "cumulative_r",
    ]

    trade_columns = [
        column
        for column in trade_columns
        if column in trades.columns
    ]

    trade_html = (
        trades[trade_columns]
        .sort_values(timestamp_column, ascending=False)
        .to_html(
            index=False,
            border=0,
            classes="data-table",
            float_format=lambda value: f"{value:.3f}",
        )
        if not trades.empty
        else (
            "<div class='notice'>"
            "No true walk-forward setups passed the current thresholds."
            "</div>"
        )
    )

    calibration_html = (
        calibration.to_html(
            index=False,
            border=0,
            classes="data-table",
            float_format=lambda value: f"{value:.3f}",
        )
        if not calibration.empty
        else "<p>No calibration rows were available.</p>"
    )

    latest_columns = [
        timestamp_column,
        "direction",
        "grade",
        "confluence_score",
        "touches",
        "probability_1r",
        "probability_2r",
        "probability_3r",
        "expected_value_3r",
        "confidence",
        "walk_forward_candidate",
        "hit_3r",
    ]

    latest_columns = [
        column
        for column in latest_columns
        if column in predictions.columns
    ]

    latest_html = (
        predictions[
            predictions["probability_3r"].notna()
        ][latest_columns]
        .tail(30)
        .sort_values(timestamp_column, ascending=False)
        .to_html(
            index=False,
            border=0,
            classes="data-table",
            float_format=lambda value: f"{value:.3f}",
        )
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LINQ V8.1 Walk-Forward Report</title>
<style>
body {{
    margin: 0;
    background: #f4f6f8;
    color: #18212b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.container {{
    max-width: 1320px;
    margin: 0 auto;
    padding: 32px 20px 60px;
}}
h1 {{
    margin-bottom: 5px;
}}
.subtitle {{
    color: #66717e;
    margin-bottom: 24px;
}}
.warning {{
    padding: 16px;
    background: #fff1d6;
    border-left: 5px solid #d89900;
    border-radius: 7px;
    margin-bottom: 22px;
}}
.success {{
    padding: 16px;
    background: #e8f6ed;
    border-left: 5px solid #278646;
    border-radius: 7px;
    margin-bottom: 22px;
}}
.notice {{
    padding: 16px;
    background: #eef2f6;
    border-left: 5px solid #697789;
    border-radius: 7px;
}}
.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 13px;
}}
.card {{
    background: white;
    padding: 17px;
    border-radius: 11px;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
}}
.label {{
    font-size: 13px;
    color: #687482;
    margin-bottom: 7px;
}}
.value {{
    font-size: 25px;
    font-weight: 700;
}}
.section {{
    background: white;
    margin-top: 20px;
    padding: 22px;
    border-radius: 11px;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
    overflow-x: auto;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th {{
    background: #edf1f4;
    text-align: left;
    padding: 10px;
    white-space: nowrap;
}}
td {{
    border-bottom: 1px solid #e7ebef;
    padding: 10px;
    white-space: nowrap;
}}
code {{
    background: #edf1f4;
    padding: 2px 6px;
    border-radius: 4px;
}}
</style>
</head>

<body>
<div class="container">
    <h1>LINQ Forex Bot V8.1</h1>
    <div class="subtitle">
        True expanding-window walk-forward report — {PAIR}
    </div>

    <div class="success">
        <strong>No future leakage:</strong>
        every probability in this report was generated by training only on
        trades that occurred earlier in time.
    </div>

    <div class="warning">
        <strong>Research only:</strong>
        the dataset is still small. A positive result must be confirmed on
        substantially more history before live execution.
    </div>

    <div class="cards">
        {metric_cards(trade_summary)}
    </div>

    <div class="section">
        <h2>1R walk-forward model</h2>
        {metrics_table(target_metrics["hit_1r"])}
    </div>

    <div class="section">
        <h2>2R walk-forward model</h2>
        {metrics_table(target_metrics["hit_2r"])}
    </div>

    <div class="section">
        <h2>3R walk-forward model</h2>
        {metrics_table(target_metrics["hit_3r"])}
    </div>

    <div class="section">
        <h2>3R probability calibration</h2>
        <p>
            This compares the probability predicted before each trade with
            the percentage that actually won.
        </p>
        {calibration_html}
    </div>

    <div class="section">
        <h2>Trades that passed all thresholds</h2>
        {trade_html}
    </div>

    <div class="section">
        <h2>Latest walk-forward predictions</h2>
        {latest_html}
    </div>

    <div class="section">
        <h2>Current selection rules</h2>
        <p>
            1R probability ≥ <strong>{MINIMUM_1R_PROBABILITY:.0%}</strong>
        </p>
        <p>
            3R probability ≥ <strong>{MINIMUM_3R_PROBABILITY:.0%}</strong>
        </p>
        <p>
            Expected value ≥ <strong>{MINIMUM_EXPECTED_VALUE_3R:.2f}R</strong>
        </p>
        <p>
            Freshness remains an input feature but receives no automatic
            scoring bonus.
        </p>
    </div>
</div>
</body>
</html>
"""


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    heading("LINQ FOREX BOT V8.1 — TRUE WALK-FORWARD ENGINE")

    merged, timestamp_column = load_data()
    features, excluded_columns = prepare_features(
        merged,
        timestamp_column,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pair:                         {PAIR}")
    print(f"Historical examples:          {len(merged)}")
    print(f"Initial training examples:    {MINIMUM_TRAIN_ROWS}")
    print(f"Walk-forward predictions:     {len(merged) - MINIMUM_TRAIN_ROWS}")
    print(f"Usable pre-entry features:    {features.shape[1]}")
    print(f"Excluded columns:             {len(excluded_columns)}")

    predictions = merged.copy()
    target_metrics: dict[str, Any] = {}

    training_sizes = None

    for target, reward_r in TARGETS.items():
        heading(f"WALK-FORWARD TARGET: {target.upper()}")

        probabilities, current_training_sizes = (
            walk_forward_predict(
                features,
                merged[target].astype(int),
            )
        )

        if training_sizes is None:
            training_sizes = current_training_sizes

        probability_column = f"probability_{reward_r}r"
        predictions[probability_column] = probabilities

        threshold = {
            "hit_1r": MINIMUM_1R_PROBABILITY,
            "hit_2r": 0.50,
            "hit_3r": MINIMUM_3R_PROBABILITY,
        }[target]

        metrics = calculate_metrics(
            merged[target],
            predictions[probability_column],
            threshold,
        )

        target_metrics[target] = metrics

        print(f"Predictions:                  {metrics.get('rows')}")
        print(
            f"Actual success rate:          "
            f"{format_pct(metrics.get('success_rate'))}"
        )
        print(
            f"Average predicted probability:"
            f" {format_pct(metrics.get('average_probability'))}"
        )
        print(
            f"Brier score:                  "
            f"{format_number(metrics.get('brier_score'))}"
        )
        print(
            f"ROC AUC:                      "
            f"{format_number(metrics.get('roc_auc'))}"
        )
        print(
            f"Precision:                    "
            f"{format_pct(metrics.get('precision'))}"
        )
        print(
            f"Recall:                       "
            f"{format_pct(metrics.get('recall'))}"
        )

    predictions["training_examples_available"] = training_sizes

    predictions["expected_value_3r"] = expected_value_3r(
        predictions["probability_3r"]
    )

    predictions["confidence"] = predictions[
        "probability_3r"
    ].map(confidence_label)

    predictions["walk_forward_candidate"] = (
        predictions["probability_1r"].notna()
        & (
            predictions["probability_1r"]
            >= MINIMUM_1R_PROBABILITY
        )
        & (
            predictions["probability_3r"]
            >= MINIMUM_3R_PROBABILITY
        )
        & (
            predictions["expected_value_3r"]
            >= MINIMUM_EXPECTED_VALUE_3R
        )
    )

    trades = build_trade_log(
        predictions,
        timestamp_column,
    )

    trade_summary = summarize_trades(
        trades,
        predictions,
        timestamp_column,
    )

    calibration = calibration_table(
        predictions["hit_3r"],
        predictions["probability_3r"],
    )

    heading("TRUE OUT-OF-SAMPLE STRATEGY RESULTS")

    print(f"Trades taken:                 {trade_summary.get('trades', 0)}")
    print(f"Winners:                      {trade_summary.get('wins', 0)}")
    print(f"Losers:                       {trade_summary.get('losses', 0)}")
    print(
        f"Win rate:                     "
        f"{format_pct(trade_summary.get('win_rate'))}"
    )
    print(
        f"Total return:                 "
        f"{format_number(trade_summary.get('total_r'))}R"
    )
    print(
        f"Expectancy:                   "
        f"{format_number(trade_summary.get('expectancy_r'))}R"
    )
    print(
        f"Profit factor:                "
        f"{format_number(trade_summary.get('profit_factor'))}"
    )
    print(
        f"Maximum drawdown:             "
        f"{format_number(trade_summary.get('maximum_drawdown_r'))}R"
    )
    print(
        f"Trades per week:              "
        f"{format_number(trade_summary.get('trades_per_week'), 2)}"
    )
    print(
        f"Weeks containing a trade:     "
        f"{format_pct(trade_summary.get('percent_weeks_with_trade'))}"
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    trades.to_csv(
        TRADES_PATH,
        index=False,
    )

    summary = {
        "pair": PAIR,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": "expanding_window_walk_forward",
        "historical_rows": int(len(merged)),
        "minimum_training_rows": MINIMUM_TRAIN_ROWS,
        "walk_forward_rows": int(
            len(merged) - MINIMUM_TRAIN_ROWS
        ),
        "feature_count": int(features.shape[1]),
        "features": features.columns.tolist(),
        "excluded_columns": excluded_columns,
        "thresholds": {
            "minimum_1r_probability": MINIMUM_1R_PROBABILITY,
            "minimum_3r_probability": MINIMUM_3R_PROBABILITY,
            "minimum_expected_value_3r": MINIMUM_EXPECTED_VALUE_3R,
        },
        "target_metrics": target_metrics,
        "trade_summary": trade_summary,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2)
    )

    HTML_PATH.write_text(
        build_html(
            predictions,
            trades,
            target_metrics,
            trade_summary,
            calibration,
            timestamp_column,
        )
    )

    heading("FILES SAVED")

    print(f"Readable report:              {HTML_PATH}")
    print(f"All predictions:              {PREDICTIONS_PATH}")
    print(f"Trades passing thresholds:    {TRADES_PATH}")
    print(f"JSON summary:                 {SUMMARY_PATH}")

    print(
        "\nOpen the report with:\n\n"
        f"open {HTML_PATH}\n"
    )

    print(divider())
    print("V8.1 completed successfully.")
    print(divider())


if __name__ == "__main__":
    main()

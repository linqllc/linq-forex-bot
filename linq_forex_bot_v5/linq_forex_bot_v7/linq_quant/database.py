from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from linq_quant.metadata import (
    classify_session,
    create_run_id,
    infer_data_source,
    infer_instrument,
    infer_strategy_version,
    infer_timeframe,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_file TEXT NOT NULL,
    data_source TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    instrument TEXT NOT NULL,
    row_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS setups (
    setup_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_file TEXT NOT NULL,
    data_source TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    instrument TEXT NOT NULL,
    timestamp TEXT,
    direction TEXT,
    session TEXT,
    weekday INTEGER,
    hour_utc INTEGER,
    setup_score REAL,
    h1_trend TEXT,
    h4_trend TEXT,
    h1_h4_agree INTEGER,
    mtf_direction_alignment REAL,
    trend_aligned INTEGER,
    break_of_structure INTEGER,
    liquidity_sweep INTEGER,
    volatility_regime TEXT,
    atr REAL,
    spread_pips REAL,
    max_favorable_r REAL,
    stopped INTEGER,
    final_r REAL,
    FOREIGN KEY(run_id) REFERENCES research_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_setups_run_id
ON setups(run_id);

CREATE INDEX IF NOT EXISTS idx_setups_source
ON setups(data_source);

CREATE INDEX IF NOT EXISTS idx_setups_strategy
ON setups(strategy_version);

CREATE INDEX IF NOT EXISTS idx_setups_instrument
ON setups(instrument);

CREATE INDEX IF NOT EXISTS idx_setups_session
ON setups(session);

CREATE INDEX IF NOT EXISTS idx_setups_timestamp
ON setups(timestamp);
"""


COLUMN_ALIASES = {
    "time": "timestamp",
    "entry_time": "timestamp",
    "score": "setup_score",
    "quality_score": "setup_score",
    "bos": "break_of_structure",
    "spread": "spread_pips",
}


DATABASE_COLUMNS = [
    "setup_id",
    "run_id",
    "source_file",
    "data_source",
    "strategy_version",
    "timeframe",
    "instrument",
    "timestamp",
    "direction",
    "session",
    "weekday",
    "hour_utc",
    "setup_score",
    "h1_trend",
    "h4_trend",
    "h1_h4_agree",
    "mtf_direction_alignment",
    "trend_aligned",
    "break_of_structure",
    "liquidity_sweep",
    "volatility_regime",
    "atr",
    "spread_pips",
    "max_favorable_r",
    "stopped",
    "final_r",
]


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)

    return connection


def _stable_setup_id(
    run_id: str,
    row: pd.Series,
) -> str:
    identity = "|".join(
        [
            run_id,
            str(row.get("instrument", "")),
            str(row.get("timestamp", "")),
            str(row.get("direction", "")),
            str(row.name),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def normalize_dataset(
    frame: pd.DataFrame,
    csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy().rename(columns=COLUMN_ALIASES)

    data_source = infer_data_source(csv_path)
    strategy_version = infer_strategy_version(csv_path)
    timeframe = infer_timeframe(result)

    instrument = (
        str(result["instrument"].dropna().iloc[0])
        if "instrument" in result.columns
        and not result["instrument"].dropna().empty
        else infer_instrument(csv_path)
    )

    run_id = create_run_id(
        csv_path,
        data_source=data_source,
        strategy_version=strategy_version,
        timeframe=timeframe,
    )

    if "timestamp" in result.columns:
        parsed_timestamps = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce",
        )

        result["timestamp"] = parsed_timestamps.astype(str)
        result["hour_utc"] = parsed_timestamps.dt.hour
        result["weekday"] = parsed_timestamps.dt.dayofweek

        # Rebuild the session field from the timestamp so old "unknown"
        # values cannot contaminate future reports.
        result["session"] = [
            classify_session(value)
            for value in parsed_timestamps
        ]
    else:
        result["timestamp"] = None
        result["session"] = "unknown"

    result["run_id"] = run_id
    result["source_file"] = str(csv_path)
    result["data_source"] = data_source
    result["strategy_version"] = strategy_version
    result["timeframe"] = timeframe
    result["instrument"] = instrument

    for column in DATABASE_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["setup_id"] = [
        _stable_setup_id(run_id, row)
        for _, row in result.iterrows()
    ]

    numeric_columns = [
        "weekday",
        "hour_utc",
        "setup_score",
        "h1_h4_agree",
        "mtf_direction_alignment",
        "trend_aligned",
        "break_of_structure",
        "liquidity_sweep",
        "atr",
        "spread_pips",
        "max_favorable_r",
        "stopped",
        "final_r",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    run_metadata = {
        "run_id": run_id,
        "source_file": str(csv_path),
        "data_source": data_source,
        "strategy_version": strategy_version,
        "timeframe": timeframe,
        "instrument": instrument,
        "row_count": len(result),
    }

    return result[DATABASE_COLUMNS], run_metadata


def import_csv(
    connection: sqlite3.Connection,
    csv_path: Path,
) -> int:
    frame = pd.read_csv(csv_path)
    normalized, run_metadata = normalize_dataset(
        frame,
        csv_path,
    )

    existing_run = connection.execute(
        """
        SELECT 1
        FROM research_runs
        WHERE run_id = ?
        """,
        (run_metadata["run_id"],),
    ).fetchone()

    if existing_run:
        return 0

    connection.execute(
        """
        INSERT INTO research_runs (
            run_id,
            source_file,
            data_source,
            strategy_version,
            timeframe,
            instrument,
            row_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_metadata["run_id"],
            run_metadata["source_file"],
            run_metadata["data_source"],
            run_metadata["strategy_version"],
            run_metadata["timeframe"],
            run_metadata["instrument"],
            run_metadata["row_count"],
        ),
    )

    normalized.to_sql(
        "incoming_setups",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        """
        INSERT OR IGNORE INTO setups (
            setup_id,
            run_id,
            source_file,
            data_source,
            strategy_version,
            timeframe,
            instrument,
            timestamp,
            direction,
            session,
            weekday,
            hour_utc,
            setup_score,
            h1_trend,
            h4_trend,
            h1_h4_agree,
            mtf_direction_alignment,
            trend_aligned,
            break_of_structure,
            liquidity_sweep,
            volatility_regime,
            atr,
            spread_pips,
            max_favorable_r,
            stopped,
            final_r
        )
        SELECT
            setup_id,
            run_id,
            source_file,
            data_source,
            strategy_version,
            timeframe,
            instrument,
            timestamp,
            direction,
            session,
            weekday,
            hour_utc,
            setup_score,
            h1_trend,
            h4_trend,
            h1_h4_agree,
            mtf_direction_alignment,
            trend_aligned,
            break_of_structure,
            liquidity_sweep,
            volatility_regime,
            atr,
            spread_pips,
            max_favorable_r,
            stopped,
            final_r
        FROM incoming_setups
        """
    )

    connection.execute("DROP TABLE incoming_setups")
    connection.commit()

    return int(len(normalized))


def import_many(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
) -> int:
    total = 0

    for path in paths:
        total += import_csv(connection, path)

    return total


def load_setups(
    connection: sqlite3.Connection,
    data_source: str | None = None,
) -> pd.DataFrame:
    query = """
    SELECT *
    FROM setups
    """
    parameters: tuple[object, ...] = ()

    if data_source:
        query += " WHERE data_source = ?"
        parameters = (data_source,)

    query += " ORDER BY timestamp"

    return pd.read_sql_query(
        query,
        connection,
        params=parameters,
    )


def load_runs(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT *
        FROM research_runs
        ORDER BY imported_at
        """,
        connection,
    )

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_SIGNAL_COLUMNS = {
    "timestamp",
    "direction",
    "entry_price",
    "stop_price",
    "target_price",
    "probability_1r",
    "gross_result_r",
    "net_result_r",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def locate_phase4_script(parent_project: Path) -> Path:
    preferred = parent_project / "run_v9_phase4_1.py"

    if preferred.exists():
        return preferred

    candidates = sorted(
        parent_project.rglob("run_v9_phase4_1.py"),
        key=lambda path: (
            len(path.parts),
            str(path),
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find run_v9_phase4_1.py beneath:\n"
            f"{parent_project}\n\n"
            "The V10 bridge will not guess or replace the validated "
            "Phase 4.1 intelligence logic."
        )

    return candidates[0]


def validate_signal_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 4.1 did not create the expected signal file:\n{path}")

    signals = pd.read_csv(path)

    missing = REQUIRED_SIGNAL_COLUMNS.difference(signals.columns)

    if missing:
        raise ValueError(
            "Phase 4.1 signal file failed its schema contract.\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(signals.columns)}"
        )

    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "entry_price",
        "stop_price",
        "target_price",
        "probability_1r",
        "gross_result_r",
        "net_result_r",
    ]

    for column in numeric_columns:
        signals[column] = pd.to_numeric(
            signals[column],
            errors="coerce",
        )

    signals["direction"] = signals["direction"].astype(str).str.strip().str.lower()

    invalid_timestamp = signals["timestamp"].isna()
    invalid_direction = ~signals["direction"].isin(["long", "short"])
    invalid_numeric = signals[numeric_columns].isna().any(axis=1)

    geometry_valid = (
        signals["direction"].eq("long")
        & signals["stop_price"].lt(signals["entry_price"])
        & signals["target_price"].gt(signals["entry_price"])
    ) | (
        signals["direction"].eq("short")
        & signals["stop_price"].gt(signals["entry_price"])
        & signals["target_price"].lt(signals["entry_price"])
    )

    invalid_geometry = ~geometry_valid

    invalid_rows = invalid_timestamp | invalid_direction | invalid_numeric | invalid_geometry

    if invalid_rows.any():
        details = signals.loc[
            invalid_rows,
            [
                "timestamp",
                "direction",
                "entry_price",
                "stop_price",
                "target_price",
            ],
        ]

        raise ValueError(
            f"Phase 4.1 produced invalid selected signals:\n{details.to_string(index=False)}"
        )

    duplicate_timestamps = int(signals["timestamp"].duplicated().sum())

    if duplicate_timestamps:
        raise ValueError(
            f"Phase 4.1 produced duplicate selected-signal timestamps: {duplicate_timestamps}"
        )

    return {
        "rows": int(len(signals)),
        "first_signal": (signals["timestamp"].min().isoformat() if len(signals) else None),
        "last_signal": (signals["timestamp"].max().isoformat() if len(signals) else None),
        "long_signals": int(signals["direction"].eq("long").sum()),
        "short_signals": int(signals["direction"].eq("short").sum()),
        "average_probability_1r": (
            float(signals["probability_1r"].mean()) if len(signals) else None
        ),
        "gross_r": float(signals["gross_result_r"].sum()),
        "net_r": float(signals["net_result_r"].sum()),
        "sha256": sha256_file(path),
    }


def run_phase4_pipeline(
    parent_project: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    script = locate_phase4_script(parent_project)

    signal_path = (
        parent_project / "reports" / "v9_phase4_1" / "EUR_USD_phase4_1_selected_trades.csv"
    )

    candle_path = parent_project / "data" / "cache" / "EUR_USD_M5.csv"

    if not candle_path.exists():
        raise FileNotFoundError(f"Candle file not found:\n{candle_path}")

    candle_hash_before = sha256_file(candle_path)
    script_hash_before = sha256_file(script)

    started_at = pd.Timestamp.now(tz="UTC")
    start_clock = time.perf_counter()

    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    duration_seconds = time.perf_counter() - start_clock
    finished_at = pd.Timestamp.now(tz="UTC")

    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stdout_path = output_manifest.parent / "phase4_1_stdout.txt"
    stderr_path = output_manifest.parent / "phase4_1_stderr.txt"

    stdout_path.write_text(
        process.stdout,
        encoding="utf-8",
    )
    stderr_path.write_text(
        process.stderr,
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "pipeline": "LINQ V10 legacy intelligence bridge",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "python_executable": sys.executable,
        "working_directory": str(script.parent),
        "phase4_script": str(script),
        "phase4_script_sha256": script_hash_before,
        "candle_file": str(candle_path),
        "candle_file_sha256": candle_hash_before,
        "signal_file": str(signal_path),
        "return_code": int(process.returncode),
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
    }

    if process.returncode != 0:
        manifest["status"] = "failed"

        output_manifest.write_text(
            json.dumps(
                manifest,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Phase 4.1 intelligence engine failed.\n\n"
            f"Return code: {process.returncode}\n"
            f"STDOUT: {stdout_path}\n"
            f"STDERR: {stderr_path}"
        )

    validation = validate_signal_file(signal_path)

    manifest["status"] = "passed"
    manifest["signal_validation"] = validation

    output_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return manifest

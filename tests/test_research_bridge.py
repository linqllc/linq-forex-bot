from pathlib import Path

import pandas as pd

from linq_platform.research.legacy_bridge import (
    sha256_file,
    validate_signal_file,
)


def test_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("LINQ", encoding="utf-8")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_signal_contract_accepts_valid_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signals.csv"

    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "direction": "long",
                "entry_price": 1.1000,
                "stop_price": 1.0990,
                "target_price": 1.1010,
                "probability_1r": 0.70,
                "gross_result_r": 1.0,
                "net_result_r": 0.80,
            }
        ]
    ).to_csv(path, index=False)

    result = validate_signal_file(path)

    assert result["rows"] == 1
    assert result["long_signals"] == 1
    assert result["short_signals"] == 0

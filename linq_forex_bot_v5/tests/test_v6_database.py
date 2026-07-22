from __future__ import annotations

from pathlib import Path

import pandas as pd

from linq_quant.database import connect, import_csv, load_setups


def test_database_import_is_idempotent(tmp_path: Path):
    csv_path = tmp_path / "EUR_USD_oanda_intelligence_dataset.csv"
    database_path = tmp_path / "research.db"

    pd.DataFrame(
        {
            "instrument": ["EUR_USD", "EUR_USD"],
            "timestamp": [
                "2026-01-01T10:00:00Z",
                "2026-01-02T10:00:00Z",
            ],
            "direction": ["long", "short"],
            "final_r": [2.0, -1.0],
        }
    ).to_csv(csv_path, index=False)

    connection = connect(database_path)

    assert import_csv(connection, csv_path) == 2
    assert import_csv(connection, csv_path) == 0
    assert len(load_setups(connection)) == 2

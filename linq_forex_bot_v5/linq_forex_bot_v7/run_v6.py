from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from linq_quant.analytics import ab_compare, performance_summary
from linq_quant.dashboard import build_dashboard
from linq_quant.database import (
    connect,
    import_many,
    load_runs,
    load_setups,
)
from linq_quant.monte_carlo import simulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LINQ Quant v6 research platform"
    )

    parser.add_argument(
        "--input",
        default="reports/v5",
        help="Directory containing v5 intelligence CSV files",
    )
    parser.add_argument(
        "--database",
        default="research/linq_quant.db",
    )
    parser.add_argument(
        "--output",
        default="reports/v6",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--data-source",
        choices=["oanda", "synthetic_demo", "all"],
        default="oanda",
        help="Data source used for static reports and Monte Carlo",
    )
    parser.add_argument(
        "--reset-database",
        action="store_true",
        help="Delete the existing database before importing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_directory = Path(args.input)
    database_path = Path(args.database)
    output_directory = Path(args.output)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.reset_database and database_path.exists():
        database_path.unlink()
        print(f"Deleted old database: {database_path}")

    csv_paths = sorted(
        input_directory.glob(
            "*_intelligence_dataset.csv"
        )
    )

    if not csv_paths:
        raise SystemExit(
            f"No intelligence datasets found in {input_directory}. "
            "Run v5 first."
        )

    connection = connect(database_path)

    print(f"Importing {len(csv_paths)} dataset(s)...")
    added = import_many(connection, csv_paths)
    print(f"Added {added} new setup row(s).")

    all_setups = load_setups(connection)

    static_source = (
        None
        if args.data_source == "all"
        else args.data_source
    )

    setups = load_setups(
        connection,
        data_source=static_source,
    )

    completed = setups.dropna(
        subset=["final_r"]
    ).copy()

    pair_summary = performance_summary(
        completed,
        ["instrument"],
    )
    session_summary = performance_summary(
        completed,
        ["session"],
    )
    weekday_summary = performance_summary(
        completed,
        ["weekday"],
    )

    if (
        "h1_h4_agree" in completed.columns
        and not completed.empty
    ):
        ab_summary = ab_compare(
            completed,
            "h1_h4_agree",
            1,
        )
    else:
        ab_summary = pd.DataFrame()

    pair_summary.to_csv(
        output_directory / "pair_performance.csv",
        index=False,
    )
    session_summary.to_csv(
        output_directory / "session_performance.csv",
        index=False,
    )
    weekday_summary.to_csv(
        output_directory / "weekday_performance.csv",
        index=False,
    )
    ab_summary.to_csv(
        output_directory / "mtf_ab_comparison.csv",
        index=False,
    )

    load_runs(connection).to_csv(
        output_directory / "research_runs.csv",
        index=False,
    )

    monte_carlo_summary = None

    if len(completed) >= 5:
        result, details = simulate(
            completed["final_r"],
            simulations=args.simulations,
        )

        monte_carlo_summary = asdict(result)

        details.to_csv(
            output_directory
            / "monte_carlo_simulations.csv",
            index=False,
        )

        (
            output_directory
            / "monte_carlo_summary.json"
        ).write_text(
            json.dumps(
                monte_carlo_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    build_dashboard(
        output_directory / "dashboard.html",
        setups=all_setups,
        pair_summary=pair_summary,
        session_summary=session_summary,
        weekday_summary=weekday_summary,
        ab_summary=ab_summary,
        monte_carlo_summary=monte_carlo_summary,
    )

    print()
    print("LINQ Quant data-quality run complete.")
    print(f"Database:  {database_path}")
    print(f"Rows:      {len(all_setups)}")
    print(f"Real rows: {(all_setups['data_source'] == 'oanda').sum()}")
    print(
        "Demo rows: "
        f"{(all_setups['data_source'] == 'synthetic_demo').sum()}"
    )
    print(f"Reports:   {output_directory}")
    print(
        f"Dashboard: {output_directory / 'dashboard.html'}"
    )


if __name__ == "__main__":
    main()

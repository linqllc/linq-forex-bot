from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent

PHASE4_PATH = (
    PARENT_DIR
    / "reports"
    / "v9_phase4_1"
    / "EUR_USD_phase4_1_selected_trades.csv"
)

V10_PATH = (
    PROJECT_DIR
    / "reports"
    / "v10"
    / "EUR_USD_v10_trade_ledger.csv"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "v10"
OUTPUT_PATH = OUTPUT_DIR / "EUR_USD_v10_parity_audit.csv"

PIP_SIZE = 0.0001


def normalized_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).dt.floor("min")


def safe_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def main() -> None:
    if not PHASE4_PATH.exists():
        raise FileNotFoundError(
            f"Phase 4.1 ledger not found:\n{PHASE4_PATH}"
        )

    if not V10_PATH.exists():
        raise FileNotFoundError(
            f"V10 Backtrader ledger not found:\n{V10_PATH}"
        )

    phase4 = pd.read_csv(PHASE4_PATH)
    v10 = pd.read_csv(V10_PATH)

    phase4["merge_timestamp"] = normalized_timestamp(
        phase4["timestamp"]
    )
    v10["merge_timestamp"] = normalized_timestamp(
        v10["signal_timestamp"]
    )

    phase4_columns = [
        "merge_timestamp",
        "timestamp",
        "direction",
        "probability_1r",
        "entry_price",
        "stop_price",
        "target_price",
        "risk_price",
        "spread_pips_at_entry",
        "spread_cost_r",
        "slippage_cost_r",
        "total_cost_r",
        "gross_result_r",
        "net_result_r",
    ]

    phase4_columns = [
        column
        for column in phase4_columns
        if column in phase4.columns
    ]

    v10_columns = [
        "merge_timestamp",
        "signal_timestamp",
        "direction",
        "entry_fill_time",
        "entry_fill_price",
        "exit_fill_time",
        "exit_fill_price",
        "exit_reason",
        "stop_price",
        "target_price",
        "risk_price",
        "realized_r_before_cost",
        "slippage_cost_r",
        "realized_r_after_slippage",
        "backtrader_gross_pnl",
        "backtrader_net_pnl",
        "bars_held",
    ]

    v10_columns = [
        column
        for column in v10_columns
        if column in v10.columns
    ]

    merged = phase4[phase4_columns].merge(
        v10[v10_columns],
        on="merge_timestamp",
        how="outer",
        suffixes=("_phase4", "_v10"),
        indicator=True,
    )

    merged["phase4_entry"] = safe_numeric(
        merged,
        "entry_price",
    )
    merged["v10_entry"] = safe_numeric(
        merged,
        "entry_fill_price",
    )

    merged["entry_difference_pips"] = (
        merged["v10_entry"]
        - merged["phase4_entry"]
    ) / PIP_SIZE

    direction_column = (
        "direction_phase4"
        if "direction_phase4" in merged
        else "direction"
    )

    direction = (
        merged[direction_column]
        .astype(str)
        .str.lower()
    )

    merged["signed_entry_difference_pips"] = np.where(
        direction.eq("long"),
        merged["v10_entry"] - merged["phase4_entry"],
        merged["phase4_entry"] - merged["v10_entry"],
    ) / PIP_SIZE

    merged["phase4_risk_pips"] = (
        safe_numeric(
            merged,
            "risk_price_phase4",
        )
        / PIP_SIZE
    )

    if merged["phase4_risk_pips"].isna().all():
        merged["phase4_risk_pips"] = (
            safe_numeric(
                merged,
                "risk_price",
            )
            / PIP_SIZE
        )

    merged["v10_risk_pips"] = (
        safe_numeric(
            merged,
            "risk_price_v10",
        )
        / PIP_SIZE
    )

    if merged["v10_risk_pips"].isna().all():
        merged["v10_risk_pips"] = (
            safe_numeric(
                merged,
                "risk_price",
            )
            / PIP_SIZE
        )

    phase4_gross = safe_numeric(
        merged,
        "gross_result_r",
    )
    phase4_net = safe_numeric(
        merged,
        "net_result_r",
    )

    v10_gross = safe_numeric(
        merged,
        "realized_r_before_cost",
    )
    v10_slippage_net = safe_numeric(
        merged,
        "realized_r_after_slippage",
    )

    phase4_spread_cost = safe_numeric(
        merged,
        "spread_cost_r",
    )

    if phase4_spread_cost.isna().all():
        spread_pips = safe_numeric(
            merged,
            "spread_pips_at_entry",
        )

        phase4_spread_cost = (
            spread_pips
            / merged["v10_risk_pips"]
        )

    merged["v10_spread_adjusted_r"] = (
        v10_slippage_net
        - phase4_spread_cost
    )

    merged["gross_r_difference"] = (
        v10_gross - phase4_gross
    )

    merged["net_r_difference_before_spread_fix"] = (
        v10_slippage_net - phase4_net
    )

    merged["net_r_difference_after_spread_fix"] = (
        merged["v10_spread_adjusted_r"]
        - phase4_net
    )

    merged["same_direction"] = True

    if (
        "direction_phase4" in merged
        and "direction_v10" in merged
    ):
        merged["same_direction"] = (
            merged["direction_phase4"]
            .astype(str)
            .str.lower()
            ==
            merged["direction_v10"]
            .astype(str)
            .str.lower()
        )

    merged["same_outcome_sign"] = (
        np.sign(phase4_gross)
        ==
        np.sign(v10_gross)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    merged.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    matched = merged[
        merged["_merge"].eq("both")
    ].copy()

    phase4_gross_total = float(
        phase4_gross.sum()
    )
    phase4_net_total = float(
        phase4_net.sum()
    )
    v10_gross_total = float(
        v10_gross.sum()
    )
    v10_slippage_total = float(
        v10_slippage_net.sum()
    )
    v10_spread_adjusted_total = float(
        merged["v10_spread_adjusted_r"].sum()
    )

    print("=" * 108)
    print(
        "LINQ V10 — BACKTRADER PARITY AUDIT"
    )
    print("=" * 108)

    print("\nMATCHING")
    print("-" * 108)
    print(
        f"Phase 4.1 trades:               "
        f"{len(phase4)}"
    )
    print(
        f"Backtrader trades:              "
        f"{len(v10)}"
    )
    print(
        f"Matched trades:                 "
        f"{len(matched)}"
    )
    print(
        f"Only in Phase 4.1:              "
        f"{int((merged['_merge'] == 'left_only').sum())}"
    )
    print(
        f"Only in Backtrader:             "
        f"{int((merged['_merge'] == 'right_only').sum())}"
    )
    print(
        f"Same outcome direction:         "
        f"{int(matched['same_outcome_sign'].sum())}"
        f" / {len(matched)}"
    )

    print("\nRESULT COMPARISON")
    print("-" * 108)
    print(
        f"Phase 4.1 gross R:              "
        f"{phase4_gross_total:+.3f}R"
    )
    print(
        f"Backtrader gross R:             "
        f"{v10_gross_total:+.3f}R"
    )
    print(
        f"Gross difference:               "
        f"{v10_gross_total - phase4_gross_total:+.3f}R"
    )
    print()
    print(
        f"Phase 4.1 net R:                "
        f"{phase4_net_total:+.3f}R"
    )
    print(
        f"Backtrader slippage-only R:     "
        f"{v10_slippage_total:+.3f}R"
    )
    print(
        f"Backtrader spread-adjusted R:   "
        f"{v10_spread_adjusted_total:+.3f}R"
    )
    print(
        f"Remaining parity difference:    "
        f"{v10_spread_adjusted_total - phase4_net_total:+.3f}R"
    )

    print("\nENTRY-FILL DIFFERENCES")
    print("-" * 108)

    valid_entry_difference = pd.to_numeric(
        matched["signed_entry_difference_pips"],
        errors="coerce",
    ).dropna()

    if len(valid_entry_difference):
        print(
            f"Average signed entry difference:"
            f" {valid_entry_difference.mean():+.3f} pips"
        )
        print(
            f"Median signed entry difference: "
            f"{valid_entry_difference.median():+.3f} pips"
        )
        print(
            f"Worst adverse entry difference: "
            f"{valid_entry_difference.max():+.3f} pips"
        )
        print(
            f"Best favorable entry difference:"
            f" {valid_entry_difference.min():+.3f} pips"
        )

    print("\nTRADE-BY-TRADE")
    print("-" * 108)

    display_columns = [
        "merge_timestamp",
        direction_column,
        "phase4_entry",
        "v10_entry",
        "signed_entry_difference_pips",
        "gross_result_r",
        "realized_r_before_cost",
        "net_result_r",
        "realized_r_after_slippage",
        "v10_spread_adjusted_r",
        "exit_reason",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in matched.columns
    ]

    print(
        matched[display_columns].to_string(
            index=False,
        )
    )

    print("\nFILE SAVED")
    print("-" * 108)
    print(OUTPUT_PATH)
    print("=" * 108)


if __name__ == "__main__":
    main()

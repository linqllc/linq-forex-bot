from pathlib import Path
import json

import pandas as pd


OUTCOMES_PATH = Path("reports/v7/EUR_USD_outcomes.csv")
SUMMARY_PATH = Path("reports/v7/EUR_USD_performance_summary.json")


def main() -> None:
    if not OUTCOMES_PATH.exists():
        SUMMARY_PATH.unlink(missing_ok=True)
        print("\nV7 EUR/USD Performance Summary\n")
        print("Total trades:          0")
        print("No H1-aligned trades are available to evaluate.")
        print("Removed any stale performance summary.")
        return

    df = pd.read_csv(OUTCOMES_PATH)

    if df.empty:
        SUMMARY_PATH.unlink(missing_ok=True)
        print("\nV7 EUR/USD Performance Summary\n")
        print("Total trades:          0")
        print("No trades are available to evaluate.")
        return

    total_trades = len(df)
    wins_1r = int(df["hit_1r"].fillna(False).astype(bool).sum())
    wins_2r = int(df["hit_2r"].fillna(False).astype(bool).sum())
    wins_3r = int(df["hit_3r"].fillna(False).astype(bool).sum())
    stopped = int(df["stop_hit"].fillna(False).astype(bool).sum())

    win_rate_1r = wins_1r / total_trades
    win_rate_2r = wins_2r / total_trades
    win_rate_3r = wins_3r / total_trades

    average_mfe_r = float(df["mfe_r"].mean())
    average_mae_r = float(df["mae_r"].mean())
    average_bars = float(df["bars_evaluated"].mean())
    average_hours = average_bars * 5 / 60

    # Fixed 3R exit model:
    # +3R for trades that reached 3R, -1R for stopped trades,
    # and 0R for unresolved trades.
    realized_r = df.apply(
        lambda row: 3.0
        if bool(row["hit_3r"])
        else (-1.0 if bool(row["stop_hit"]) else 0.0),
        axis=1,
    )

    total_r = float(realized_r.sum())
    average_r = float(realized_r.mean())

    gross_profit_r = float(realized_r[realized_r > 0].sum())
    gross_loss_r = abs(float(realized_r[realized_r < 0].sum()))

    profit_factor = (
        gross_profit_r / gross_loss_r
        if gross_loss_r > 0
        else None
    )

    summary = {
        "total_trades": total_trades,
        "wins_1r": wins_1r,
        "wins_2r": wins_2r,
        "wins_3r": wins_3r,
        "stopped_trades": stopped,
        "win_rate_1r": round(win_rate_1r, 4),
        "win_rate_2r": round(win_rate_2r, 4),
        "win_rate_3r": round(win_rate_3r, 4),
        "average_mfe_r": round(average_mfe_r, 4),
        "average_mae_r": round(average_mae_r, 4),
        "average_bars_to_exit": round(average_bars, 2),
        "average_hours_to_exit": round(average_hours, 2),
        "total_realized_r": round(total_r, 2),
        "average_realized_r": round(average_r, 2),
        "profit_factor": (
            round(profit_factor, 2)
            if profit_factor is not None
            else None
        ),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\nV7 EUR/USD Performance Summary\n")
    print(f"Total trades:          {total_trades}")
    print(f"1R win rate:           {win_rate_1r:.1%}")
    print(f"2R win rate:           {win_rate_2r:.1%}")
    print(f"3R win rate:           {win_rate_3r:.1%}")
    print(f"Stopped trades:        {stopped}")
    print(f"Average MFE:           {average_mfe_r:.2f}R")
    print(f"Average MAE:           {average_mae_r:.2f}R")
    print(f"Average exit time:     {average_hours:.2f} hours")
    print(f"Total realized return: {total_r:.2f}R")
    print(f"Average expectancy:    {average_r:.2f}R per trade")

    if profit_factor is None:
        print("Profit factor:         No losing trades")
    else:
        print(f"Profit factor:         {profit_factor:.2f}")

    print(f"\nSaved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

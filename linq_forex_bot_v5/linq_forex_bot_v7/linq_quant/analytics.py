from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    usable = frame.copy()
    usable["final_r"] = pd.to_numeric(
        usable["final_r"],
        errors="coerce",
    )
    usable = usable.dropna(subset=["final_r"])

    if usable.empty:
        return pd.DataFrame()

    grouped = usable.groupby(
        group_columns,
        dropna=False,
    )["final_r"]

    summary = grouped.agg(
        trades="count",
        net_r="sum",
        expectancy_r="mean",
        median_r="median",
    ).reset_index()

    win_rates = grouped.apply(
        lambda series: float((series > 0).mean())
    ).reset_index(name="win_rate")

    gross_profit = grouped.apply(
        lambda series: float(series[series > 0].sum())
    ).reset_index(name="gross_profit_r")

    gross_loss = grouped.apply(
        lambda series: float(abs(series[series < 0].sum()))
    ).reset_index(name="gross_loss_r")

    summary = summary.merge(
        win_rates,
        on=group_columns,
        how="left",
    )
    summary = summary.merge(
        gross_profit,
        on=group_columns,
        how="left",
    )
    summary = summary.merge(
        gross_loss,
        on=group_columns,
        how="left",
    )

    summary["profit_factor"] = np.where(
        summary["gross_loss_r"] > 0,
        summary["gross_profit_r"] / summary["gross_loss_r"],
        np.nan,
    )

    return summary.sort_values(
        ["expectancy_r", "trades"],
        ascending=[False, False],
    )


def ab_compare(
    frame: pd.DataFrame,
    feature: str,
    enabled_value: object = 1,
) -> pd.DataFrame:
    if feature not in frame.columns:
        raise KeyError(f"Missing A/B feature: {feature}")

    result = frame.copy()
    result["variant"] = np.where(
        result[feature] == enabled_value,
        f"{feature}=enabled",
        f"{feature}=disabled",
    )

    return performance_summary(result, ["variant"])

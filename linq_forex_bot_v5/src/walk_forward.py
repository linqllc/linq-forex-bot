from __future__ import annotations
import pandas as pd
from src.optimizer import optimize_setups, evaluate_ratio, chronological_split


def walk_forward_report(setups: pd.DataFrame, ratios: list[float], config: dict) -> tuple[pd.DataFrame, dict]:
    oc=config["optimizer"]
    train, validation, test = chronological_split(setups, oc["train_fraction"], oc["validation_fraction"])
    train_results, selected = optimize_setups(train, ratios, oc["minimum_trades"])
    chosen=selected["best_overall_ratio"]
    rows=[]
    for name, frame in (("train",train),("validation",validation),("test",test)):
        row=evaluate_ratio(frame,chosen); row["split"]=name; rows.append(row)
    summary={"selected_ratio_from_train":chosen,"train_setups":len(train),"validation_setups":len(validation),"test_setups":len(test)}
    return pd.DataFrame(rows), summary

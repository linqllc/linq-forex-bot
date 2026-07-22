from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .config import ResearchConfig
from .data import load_candles
from .features import add_market_features
from .labels import label_outcomes
from .research import (
    chronological_split,
    performance_summary,
    rank_single_features,
    search_rule_combinations,
)
from .setups import detect_candidate_setups


class ForexResearchEngine:
    def __init__(self, config: ResearchConfig):
        self.config = config

    def run(self, csv_path: str | Path, report_dir: str | Path) -> dict:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        candles = load_candles(csv_path, self.config.timestamp_column)
        featured = add_market_features(
            candles,
            atr_period=self.config.atr_period,
            ema_fast=self.config.ema_fast,
            ema_slow=self.config.ema_slow,
        )
        setups = detect_candidate_setups(featured, self.config)
        labeled = label_outcomes(featured, setups, self.config)

        if labeled.empty:
            raise RuntimeError(
                "No candidate setups were generated. Lower displacement thresholds or inspect the data."
            )

        train, test = chronological_split(labeled, self.config.train_fraction)
        ranking = rank_single_features(train)
        rules = search_rule_combinations(train, test, ranking, self.config)

        setup_path = report_dir / f"{self.config.instrument}_v10_setups.csv"
        rank_path = report_dir / f"{self.config.instrument}_v10_feature_ranking.csv"
        rules_path = report_dir / f"{self.config.instrument}_v10_rule_validation.csv"
        summary_path = report_dir / f"{self.config.instrument}_v10_summary.json"

        labeled.to_csv(setup_path, index=False)
        ranking.to_csv(rank_path, index=False)
        rules.to_csv(rules_path, index=False)

        summary = {
            "instrument": self.config.instrument,
            "candle_rows": len(candles),
            "candidate_setups": len(labeled),
            "train_setups": len(train),
            "test_setups": len(test),
            "baseline_all": performance_summary(labeled),
            "baseline_train": performance_summary(train),
            "baseline_test": performance_summary(test),
            "best_validated_rule": rules.iloc[0].to_dict() if not rules.empty else None,
            "reports": {
                "setups": str(setup_path),
                "feature_ranking": str(rank_path),
                "rule_validation": str(rules_path),
            },
            "warning": (
                "Research output is not proof of future profitability. "
                "Rules must be locked and validated on additional unseen data before paper trading."
            ),
        }

        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        summary["reports"]["summary"] = str(summary_path)
        return summary

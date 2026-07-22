from __future__ import annotations

from pathlib import Path
import json

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
from .robustness import retrace_sensitivity, walk_forward_validate
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
            raise RuntimeError("No candidate setups were generated.")

        train, test = chronological_split(
            labeled,
            self.config.train_fraction,
        )

        ranking = rank_single_features(train)

        rules = search_rule_combinations(
            train,
            test,
            ranking,
            self.config,
        )

        sensitivity = retrace_sensitivity(
            train,
            test,
            center=0.25,
        )

        best_rule = rules.iloc[0].to_dict() if not rules.empty else None

        walk_forward = None

        if best_rule is not None:
            walk_forward = walk_forward_validate(
                labeled,
                str(best_rule["rules"]),
                folds=self.config.walk_forward_folds,
                min_fold_trades=self.config.min_fold_trades,
            )

        prefix = f"{self.config.instrument}_v10"

        setup_path = report_dir / f"{prefix}_setups.csv"
        ranking_path = report_dir / f"{prefix}_feature_ranking.csv"
        rules_path = report_dir / f"{prefix}_rule_validation.csv"
        sensitivity_path = report_dir / f"{prefix}_retrace_sensitivity.csv"
        walk_forward_path = report_dir / f"{prefix}_walk_forward.csv"
        summary_path = report_dir / f"{prefix}_summary.json"

        labeled.to_csv(setup_path, index=False)
        ranking.to_csv(ranking_path, index=False)
        rules.to_csv(rules_path, index=False)
        sensitivity.to_csv(sensitivity_path, index=False)

        if walk_forward is not None:
            walk_forward.to_csv(walk_forward_path, index=False)

        walk_forward_summary = None

        if walk_forward is not None and not walk_forward.empty:
            valid_folds = walk_forward[
                walk_forward["passes_min_trades"]
            ]

            walk_forward_summary = {
                "folds": int(len(walk_forward)),
                "valid_folds": int(len(valid_folds)),
                "profitable_valid_folds": int(
                    (valid_folds["expectancy_r"] > 0).sum()
                ),
                "median_expectancy_r": (
                    float(valid_folds["expectancy_r"].median())
                    if not valid_folds.empty
                    else None
                ),
                "worst_expectancy_r": (
                    float(valid_folds["expectancy_r"].min())
                    if not valid_folds.empty
                    else None
                ),
            }

        summary = {
            "engine_version": "10.1-robustness",
            "instrument": self.config.instrument,
            "candle_rows": len(candles),
            "candidate_setups": len(labeled),
            "train_setups": len(train),
            "test_setups": len(test),
            "minimum_rule_samples": {
                "train": self.config.min_train_trades,
                "test": self.config.min_test_trades,
            },
            "baseline_all": performance_summary(labeled),
            "baseline_train": performance_summary(train),
            "baseline_test": performance_summary(test),
            "best_validated_rule": best_rule,
            "walk_forward_summary": walk_forward_summary,
            "reports": {
                "setups": str(setup_path),
                "feature_ranking": str(ranking_path),
                "rule_validation": str(rules_path),
                "retrace_sensitivity": str(sensitivity_path),
                "walk_forward": (
                    str(walk_forward_path)
                    if walk_forward is not None
                    else None
                ),
                "summary": str(summary_path),
            },
            "warning": (
                "Research output is not proof of future profitability. "
                "Lock rules and validate on additional unseen data before "
                "paper trading."
            ),
        }

        summary_path.write_text(
            json.dumps(summary, indent=2, default=str)
        )

        return summary

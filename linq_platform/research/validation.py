from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationLimits:
    maximum_net_parity_difference_r: float = 0.10
    maximum_gross_parity_difference_r: float = 0.10
    minimum_outcome_match_rate: float = 1.00
    require_all_signals_submitted: bool = True
    require_all_submitted_trades_closed: bool = True
    allow_skipped_signals: bool = False


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return number


def validate_backtest(
    summary: dict[str, Any],
    limits: ValidationLimits | None = None,
) -> dict[str, Any]:
    limits = limits or ValidationLimits()

    parity = summary.get("parity", {})
    net = summary.get("net_performance", {})

    signals_loaded = int(summary.get("signals_loaded", 0))
    signals_submitted = int(summary.get("signals_submitted", 0))
    signals_skipped = int(summary.get("signals_skipped", 0))
    completed_trades = int(net.get("trades", 0))

    matched_trades = int(parity.get("matched_trades", 0))
    same_outcome_trades = int(
        parity.get("same_outcome_trades", 0)
    )

    outcome_match_rate = (
        same_outcome_trades / matched_trades
        if matched_trades > 0
        else 0.0
    )

    gross_difference = abs(
        _finite_float(
            parity.get("gross_parity_difference_r")
        )
    )

    net_difference = abs(
        _finite_float(
            parity.get("net_parity_difference_r")
        )
    )

    checks = {
        "signals_exist": {
            "passed": signals_loaded > 0,
            "observed": signals_loaded,
            "required": "> 0",
        },
        "all_signals_submitted": {
            "passed": (
                signals_submitted == signals_loaded
                if limits.require_all_signals_submitted
                else True
            ),
            "observed": signals_submitted,
            "required": signals_loaded,
        },
        "skipped_signals_allowed": {
            "passed": (
                signals_skipped == 0
                if not limits.allow_skipped_signals
                else True
            ),
            "observed": signals_skipped,
            "required": 0,
        },
        "all_submitted_trades_closed": {
            "passed": (
                completed_trades == signals_submitted
                if limits.require_all_submitted_trades_closed
                else True
            ),
            "observed": completed_trades,
            "required": signals_submitted,
        },
        "outcome_match_rate": {
            "passed": (
                outcome_match_rate
                >= limits.minimum_outcome_match_rate
            ),
            "observed": outcome_match_rate,
            "required": limits.minimum_outcome_match_rate,
        },
        "gross_parity_difference": {
            "passed": (
                gross_difference
                <= limits.maximum_gross_parity_difference_r
            ),
            "observed": gross_difference,
            "required": (
                limits.maximum_gross_parity_difference_r
            ),
        },
        "net_parity_difference": {
            "passed": (
                net_difference
                <= limits.maximum_net_parity_difference_r
            ),
            "observed": net_difference,
            "required": limits.maximum_net_parity_difference_r,
        },
    }

    failed_checks = [
        name
        for name, check in checks.items()
        if not check["passed"]
    ]

    return {
        "status": "PASSED" if not failed_checks else "FAILED",
        "passed": not failed_checks,
        "limits": asdict(limits),
        "checks": checks,
        "failed_checks": failed_checks,
        "outcome_match_rate": outcome_match_rate,
        "gross_parity_difference_r_absolute": gross_difference,
        "net_parity_difference_r_absolute": net_difference,
    }


def print_validation_report(
    validation: dict[str, Any],
) -> None:
    print("\nRESEARCH VALIDATION GATE")
    print("-" * 108)

    for name, check in validation["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"

        label = name.replace("_", " ").title()

        observed = check["observed"]
        required = check["required"]

        if isinstance(observed, float):
            observed_text = f"{observed:.6f}"
        else:
            observed_text = str(observed)

        if isinstance(required, float):
            required_text = f"{required:.6f}"
        else:
            required_text = str(required)

        print(
            f"{status:<6}"
            f"{label:<39}"
            f"Observed: {observed_text:<14}"
            f"Required: {required_text}"
        )

    print("-" * 108)
    print(
        f"Validation status:              "
        f"{validation['status']}"
    )

from __future__ import annotations


def sample_confidence_label(trades: int) -> str:
    if trades < 30:
        return "very_low"

    if trades < 100:
        return "low"

    if trades < 300:
        return "moderate"

    if trades < 1000:
        return "good"

    return "strong"


def sample_warning(trades: int) -> str:
    label = sample_confidence_label(trades)

    messages = {
        "very_low": (
            "Very small sample. Do not use this result for live-trading "
            "decisions."
        ),
        "low": (
            "Small sample. Treat the result as preliminary."
        ),
        "moderate": (
            "Moderate sample. Additional out-of-sample testing is still "
            "required."
        ),
        "good": (
            "Good research sample, but robustness testing remains necessary."
        ),
        "strong": (
            "Large sample. Continue checking regime stability and data quality."
        ),
    }

    return messages[label]

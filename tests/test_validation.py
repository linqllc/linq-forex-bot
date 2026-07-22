from linq_platform.research.validation import (
    ValidationLimits,
    validate_backtest,
)


def valid_summary():
    return {
        "signals_loaded": 5,
        "signals_submitted": 5,
        "signals_skipped": 0,
        "net_performance": {
            "trades": 5,
        },
        "parity": {
            "matched_trades": 5,
            "same_outcome_trades": 5,
            "gross_parity_difference_r": -0.056,
            "net_parity_difference_r": -0.054,
        },
    }


def test_valid_backtest_passes():
    result = validate_backtest(
        valid_summary(),
        ValidationLimits(),
    )

    assert result["passed"] is True
    assert result["status"] == "PASSED"
    assert result["failed_checks"] == []


def test_large_parity_difference_fails():
    summary = valid_summary()

    summary["parity"][
        "net_parity_difference_r"
    ] = 0.50

    result = validate_backtest(
        summary,
        ValidationLimits(
            maximum_net_parity_difference_r=0.10
        ),
    )

    assert result["passed"] is False
    assert (
        "net_parity_difference"
        in result["failed_checks"]
    )


def test_skipped_signal_fails():
    summary = valid_summary()

    summary["signals_skipped"] = 1
    summary["signals_submitted"] = 4
    summary["net_performance"]["trades"] = 4

    result = validate_backtest(
        summary,
        ValidationLimits(),
    )

    assert result["passed"] is False
    assert (
        "skipped_signals_allowed"
        in result["failed_checks"]
    )

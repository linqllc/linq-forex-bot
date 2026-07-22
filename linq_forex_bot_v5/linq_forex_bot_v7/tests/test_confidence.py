from linq_quant.confidence import (
    sample_confidence_label,
    sample_warning,
)


def test_confidence_levels():
    assert sample_confidence_label(10) == "very_low"
    assert sample_confidence_label(50) == "low"
    assert sample_confidence_label(200) == "moderate"
    assert sample_confidence_label(500) == "good"
    assert sample_confidence_label(1500) == "strong"


def test_warning_is_returned():
    assert "sample" in sample_warning(20).lower()

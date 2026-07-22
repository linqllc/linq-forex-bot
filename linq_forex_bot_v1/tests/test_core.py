from src.demo_data import generate_demo_data
from src.indicators import add_indicators
from src.session import add_session_columns, calculate_opening_ranges


def test_demo_pipeline():
    frame = generate_demo_data(days=5)
    frame = add_session_columns(frame, "America/New_York")
    frame = add_indicators(frame)
    frame = calculate_opening_ranges(frame, "09:30", "09:45")
    assert len(frame) > 0
    assert "atr" in frame.columns
    assert "opening_range_high" in frame.columns

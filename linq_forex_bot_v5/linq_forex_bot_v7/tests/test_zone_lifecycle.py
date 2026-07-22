import pandas as pd

from src.automatic_supply_demand import (
    _count_zone_touches,
    _find_zone_invalidation_index,
    _zone_is_invalidated,
)


def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_demand_zone_invalidation_uses_close_below_bottom() -> None:
    row = pd.Series(
        {
            "mid_close": 0.99,
        }
    )

    assert _zone_is_invalidated(
        row,
        zone_bottom=1.00,
        zone_top=1.02,
        zone_type="demand",
    )


def test_supply_zone_invalidation_uses_close_above_top() -> None:
    row = pd.Series(
        {
            "mid_close": 1.03,
        }
    )

    assert _zone_is_invalidated(
        row,
        zone_bottom=1.00,
        zone_top=1.02,
        zone_type="supply",
    )


def test_finds_first_invalidation_index() -> None:
    frame = _frame(
        [
            {
                "mid_low": 1.01,
                "mid_high": 1.03,
                "mid_close": 1.01,
            },
            {
                "mid_low": 0.99,
                "mid_high": 1.01,
                "mid_close": 1.00,
            },
            {
                "mid_low": 0.97,
                "mid_high": 1.00,
                "mid_close": 0.98,
            },
            {
                "mid_low": 0.95,
                "mid_high": 0.99,
                "mid_close": 0.96,
            },
        ]
    )

    result = _find_zone_invalidation_index(
        frame,
        start_index=0,
        zone_top=1.02,
        zone_bottom=0.99,
        zone_type="demand",
    )

    assert result == 2


def test_touch_count_stops_before_invalidation() -> None:
    frame = _frame(
        [
            {
                "mid_low": 1.03,
                "mid_high": 1.04,
                "mid_close": 1.03,
            },
            {
                "mid_low": 1.00,
                "mid_high": 1.03,
                "mid_close": 1.01,
            },
            {
                "mid_low": 1.01,
                "mid_high": 1.03,
                "mid_close": 1.02,
            },
            {
                "mid_low": 1.03,
                "mid_high": 1.04,
                "mid_close": 1.03,
            },
            {
                "mid_low": 0.98,
                "mid_high": 1.01,
                "mid_close": 0.98,
            },
            {
                "mid_low": 1.00,
                "mid_high": 1.02,
                "mid_close": 1.01,
            },
        ]
    )

    touches = _count_zone_touches(
        frame,
        start_index=0,
        zone_top=1.02,
        zone_bottom=0.99,
        end_index=4,
    )

    assert touches == 1

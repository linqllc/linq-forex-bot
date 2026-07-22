import pandas as pd
import pytest

from src.research_v10.replay import (
    apply_rule,
    build_decision_log,
    parse_rule,
    ReplayConfig,
)


def test_parse_rule():
    parsed = parse_rule(
        "retrace_fraction le 0.25 AND session eq london_open"
    )

    assert parsed == [
        ("retrace_fraction", "le", "0.25"),
        ("session", "eq", "london_open"),
    ]


def test_apply_rule():
    dataframe = pd.DataFrame(
        {
            "retrace_fraction": [0.20, 0.30, 0.20],
            "session": [
                "london_open",
                "london_open",
                "new_york_overlap",
            ],
        }
    )

    mask = apply_rule(
        dataframe,
        "retrace_fraction le 0.25 AND session eq london_open",
    )

    assert mask.tolist() == [True, False, False]


def test_replay_decision_log():
    setups = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T08:00:00Z",
                "2026-01-01T09:00:00Z",
            ],
            "retrace_fraction": [0.20, 0.40],
            "session": ["london_open", "london_open"],
            "outcome_r": [1.5, -1.0],
        }
    )

    decisions = build_decision_log(
        setups,
        ReplayConfig(
            rule=(
                "retrace_fraction le 0.25 "
                "AND session eq london_open"
            )
        ),
    )

    assert decisions["accepted"].tolist() == [True, False]


def test_future_label_cannot_be_used_in_rule():
    setups = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T08:00:00Z"],
            "outcome_r": [1.5],
        }
    )

    with pytest.raises(ValueError):
        build_decision_log(
            setups,
            ReplayConfig(rule="outcome_r ge 1.0"),
        )

import pandas as pd
import pytest

from backend.app.model.rolling_features import (
    build_rolling_features,
)


@pytest.fixture
def sample_matches():
    return pd.DataFrame(
        [
            {
                "Date": "01/01/2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 0,
                "HS": 14,
                "AS": 7,
                "HST": 6,
                "AST": 2,
            },
            {
                "Date": "08/01/2025",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Liverpool",
                "FTHG": 1,
                "FTAG": 1,
                "HS": 10,
                "AS": 12,
                "HST": 4,
                "AST": 5,
            },
            {
                "Date": "15/01/2025",
                "HomeTeam": "Liverpool",
                "AwayTeam": "Arsenal",
                "FTHG": 3,
                "FTAG": 1,
                "HS": 16,
                "AS": 8,
                "HST": 7,
                "AST": 3,
            },
            {
                "Date": "22/01/2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Liverpool",
                "FTHG": 2,
                "FTAG": 2,
                "HS": 13,
                "AS": 11,
                "HST": 5,
                "AST": 4,
            },
        ]
    )


def test_returns_same_number_of_matches(sample_matches):
    result = build_rolling_features(sample_matches)

    assert len(result) == len(sample_matches)


def test_first_match_uses_no_team_history(sample_matches):
    result = build_rolling_features(sample_matches)

    first_match = result.iloc[0]

    assert first_match["home_matches_available"] == 0
    assert first_match["away_matches_available"] == 0


def test_current_match_does_not_leak_into_features(
    sample_matches,
):
    result = build_rolling_features(sample_matches)

    arsenal_second_match = result.iloc[2]

    # Arsenal scored two goals in its first match.
    # The current match's one Arsenal goal must not be included.
    assert arsenal_second_match[
        "away_recent_goals_for"
    ] == pytest.approx(2.0)

    assert arsenal_second_match[
        "away_recent_goals_against"
    ] == pytest.approx(0.0)


def test_points_are_based_only_on_previous_matches(
    sample_matches,
):
    result = build_rolling_features(sample_matches)

    arsenal_second_match = result.iloc[2]

    # Arsenal won its only previous match.
    assert arsenal_second_match[
        "away_recent_points"
    ] == pytest.approx(3.0)


def test_recent_shots_are_created(sample_matches):
    result = build_rolling_features(sample_matches)

    arsenal_second_match = result.iloc[2]

    assert arsenal_second_match[
        "away_recent_shots"
    ] == pytest.approx(14.0)

    assert arsenal_second_match[
        "away_recent_shots_on_target"
    ] == pytest.approx(6.0)


def test_window_limits_number_of_previous_matches():
    matches = pd.DataFrame(
        [
            {
                "Date": f"{day:02d}/01/2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": f"Opponent {day}",
                "FTHG": goals,
                "FTAG": 0,
            }
            for day, goals in enumerate(
                [1, 2, 3, 4],
                start=1,
            )
        ]
    )

    result = build_rolling_features(
        matches,
        window=2,
    )

    final_match = result.iloc[3]

    # Only Arsenal's previous two results, 2 and 3,
    # should be included.
    assert final_match[
        "home_recent_goals_for"
    ] == pytest.approx(2.5)


def test_invalid_window_raises_error(sample_matches):
    with pytest.raises(ValueError):
        build_rolling_features(
            sample_matches,
            window=0,
        )
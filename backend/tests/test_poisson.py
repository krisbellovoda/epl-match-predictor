import pytest

from backend.app.model.poisson import predict_match


def test_match_result_probabilities_sum_to_one():
    prediction = predict_match(
        home_expected_goals=1.75,
        away_expected_goals=0.90,
    )

    match_result = prediction["match_result"]

    total = (
        match_result["home_win"]
        + match_result["draw"]
        + match_result["away_win"]
    )

    assert total == pytest.approx(1.0)


def test_goal_market_probabilities_sum_to_one():
    prediction = predict_match(
        home_expected_goals=1.75,
        away_expected_goals=0.90,
    )

    total_goals = prediction["total_goals"]

    assert (
        total_goals["over_2_5"]
        + total_goals["under_2_5"]
    ) == pytest.approx(1.0)


def test_btts_probabilities_sum_to_one():
    prediction = predict_match(
        home_expected_goals=1.75,
        away_expected_goals=0.90,
    )

    btts = prediction["both_teams_to_score"]

    assert (
        btts["yes"] + btts["no"]
    ) == pytest.approx(1.0)


def test_equal_expected_goals_produce_equal_win_probabilities():
    prediction = predict_match(
        home_expected_goals=1.25,
        away_expected_goals=1.25,
    )

    match_result = prediction["match_result"]

    assert match_result["home_win"] == pytest.approx(
        match_result["away_win"]
    )


def test_higher_home_xg_favors_home_team():
    prediction = predict_match(
        home_expected_goals=2.50,
        away_expected_goals=0.50,
    )

    match_result = prediction["match_result"]

    assert (
        match_result["home_win"]
        > match_result["away_win"]
    )


def test_expected_goals_must_be_positive():
    with pytest.raises(ValueError):
        predict_match(
            home_expected_goals=0,
            away_expected_goals=1.0,
        )


def test_returns_five_top_scorelines():
    prediction = predict_match(
        home_expected_goals=1.75,
        away_expected_goals=0.90,
    )

    assert len(prediction["top_scorelines"]) == 5
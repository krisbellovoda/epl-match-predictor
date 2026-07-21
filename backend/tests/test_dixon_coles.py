import numpy as np
import pytest

from backend.app.model.poisson import (
    build_score_matrix,
    predict_match,
)


def test_score_matrix_sums_to_one():
    matrix = build_score_matrix(
        home_expected_goals=1.7,
        away_expected_goals=1.1,
        dixon_coles_rho=-0.08,
    )

    assert matrix.sum() == pytest.approx(1.0)


def test_zero_rho_matches_independent_poisson():
    original = predict_match(
        home_expected_goals=1.7,
        away_expected_goals=1.1,
    )

    zero_adjustment = predict_match(
        home_expected_goals=1.7,
        away_expected_goals=1.1,
        dixon_coles_rho=0.0,
    )

    assert (
        zero_adjustment["match_result"]
        == original["match_result"]
    )

    assert (
        zero_adjustment["total_goals"]
        == original["total_goals"]
    )


def test_negative_rho_increases_low_score_draws():
    independent = build_score_matrix(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        dixon_coles_rho=0.0,
    )

    adjusted = build_score_matrix(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        dixon_coles_rho=-0.08,
    )

    assert adjusted[0, 0] > independent[0, 0]
    assert adjusted[1, 1] > independent[1, 1]


def test_dixon_coles_changes_match_probabilities():
    independent = predict_match(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        dixon_coles_rho=0.0,
    )

    adjusted = predict_match(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        dixon_coles_rho=-0.08,
    )

    assert (
        adjusted["match_result"]["draw"]
        != pytest.approx(
            independent["match_result"]["draw"]
        )
    )


def test_model_metadata_identifies_dixon_coles():
    prediction = predict_match(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        dixon_coles_rho=-0.08,
    )

    assert prediction["model"]["name"] == "dixon_coles"
    assert (
        prediction["model"]["dixon_coles_rho"]
        == pytest.approx(-0.08)
    )


def test_invalid_expected_goals_raise_error():
    with pytest.raises(ValueError):
        build_score_matrix(
            home_expected_goals=0,
            away_expected_goals=1.1,
        )


def test_invalid_correction_factor_raises_error():
    with pytest.raises(ValueError):
        build_score_matrix(
            home_expected_goals=2.0,
            away_expected_goals=2.0,
            dixon_coles_rho=-1.0,
        )


def test_matrix_contains_no_negative_probabilities():
    matrix = build_score_matrix(
        home_expected_goals=1.6,
        away_expected_goals=1.0,
        dixon_coles_rho=-0.08,
    )

    assert np.all(matrix >= 0)
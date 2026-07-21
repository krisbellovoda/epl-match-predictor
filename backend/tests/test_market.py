import pytest

from backend.app.model.market import (
    compare_two_way_market,
)


def test_no_vig_probabilities_sum_to_one():
    comparison = compare_two_way_market(
        model_over_probability=0.55,
        over_decimal_odds=1.95,
        under_decimal_odds=1.90,
    )

    no_vig_market = comparison["no_vig_market"]

    total = (
        no_vig_market["over_2_5"]
        + no_vig_market["under_2_5"]
    )

    assert total == pytest.approx(1.0)


def test_equal_odds_produce_equal_market_probabilities():
    comparison = compare_two_way_market(
        model_over_probability=0.55,
        over_decimal_odds=1.90,
        under_decimal_odds=1.90,
    )

    no_vig_market = comparison["no_vig_market"]

    assert no_vig_market[
        "over_2_5"
    ] == pytest.approx(0.5)

    assert no_vig_market[
        "under_2_5"
    ] == pytest.approx(0.5)


def test_positive_expected_value():
    comparison = compare_two_way_market(
        model_over_probability=0.60,
        over_decimal_odds=2.00,
        under_decimal_odds=1.80,
    )

    assert comparison[
        "expected_value"
    ]["over_2_5"] == pytest.approx(0.20)


def test_invalid_decimal_odds_are_rejected():
    with pytest.raises(ValueError):
        compare_two_way_market(
            model_over_probability=0.55,
            over_decimal_odds=1.00,
            under_decimal_odds=1.90,
        )


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        compare_two_way_market(
            model_over_probability=1.20,
            over_decimal_odds=1.90,
            under_decimal_odds=1.90,
        )
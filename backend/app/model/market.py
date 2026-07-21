def compare_two_way_market(
    model_over_probability: float,
    over_decimal_odds: float,
    under_decimal_odds: float,
) -> dict:
    """
    Compare model probabilities with a two-way sportsbook market.

    The sportsbook margin is removed before calculating edges.
    """

    if not 0 <= model_over_probability <= 1:
        raise ValueError(
            "Model probability must be between 0 and 1."
        )

    if over_decimal_odds <= 1:
        raise ValueError(
            "Over odds must be greater than 1."
        )

    if under_decimal_odds <= 1:
        raise ValueError(
            "Under odds must be greater than 1."
        )

    model_under_probability = (
        1 - model_over_probability
    )

    raw_over_probability = 1 / over_decimal_odds
    raw_under_probability = 1 / under_decimal_odds

    sportsbook_margin = (
        raw_over_probability
        + raw_under_probability
        - 1
    )

    raw_probability_total = (
        raw_over_probability
        + raw_under_probability
    )

    market_over_probability = (
        raw_over_probability
        / raw_probability_total
    )

    market_under_probability = (
        raw_under_probability
        / raw_probability_total
    )

    over_edge = (
        model_over_probability
        - market_over_probability
    )

    under_edge = (
        model_under_probability
        - market_under_probability
    )

    over_expected_value = (
        model_over_probability
        * over_decimal_odds
        - 1
    )

    under_expected_value = (
        model_under_probability
        * under_decimal_odds
        - 1
    )

    return {
        "sportsbook_margin": float(
            sportsbook_margin
        ),
        "no_vig_market": {
            "over_2_5": float(
                market_over_probability
            ),
            "under_2_5": float(
                market_under_probability
            ),
        },
        "model": {
            "over_2_5": float(
                model_over_probability
            ),
            "under_2_5": float(
                model_under_probability
            ),
        },
        "edge": {
            "over_2_5": float(over_edge),
            "under_2_5": float(under_edge),
        },
        "expected_value": {
            "over_2_5": float(
                over_expected_value
            ),
            "under_2_5": float(
                under_expected_value
            ),
        },
    }
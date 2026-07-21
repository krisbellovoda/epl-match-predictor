import numpy as np
from scipy.stats import poisson


def _validate_inputs(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int,
) -> None:
    if home_expected_goals <= 0:
        raise ValueError(
            "Home expected goals must be greater than zero."
        )

    if away_expected_goals <= 0:
        raise ValueError(
            "Away expected goals must be greater than zero."
        )

    if max_goals < 1:
        raise ValueError(
            "max_goals must be at least 1."
        )


def _dixon_coles_factors(
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
) -> dict[tuple[int, int], float]:
    """
    Return Dixon-Coles correction factors for low scores.

    The correction only affects:
    - 0-0
    - 0-1
    - 1-0
    - 1-1

    rho=0 produces the original independent Poisson model.
    """

    return {
        (0, 0): (
            1
            - home_expected_goals
            * away_expected_goals
            * rho
        ),
        (0, 1): (
            1
            + home_expected_goals * rho
        ),
        (1, 0): (
            1
            + away_expected_goals * rho
        ),
        (1, 1): 1 - rho,
    }


def build_score_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
    dixon_coles_rho: float = 0.0,
) -> np.ndarray:
    """
    Build a normalized score-probability matrix.

    Rows represent home goals.
    Columns represent away goals.
    """

    _validate_inputs(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        max_goals=max_goals,
    )

    goal_values = np.arange(max_goals + 1)

    home_goal_probabilities = poisson.pmf(
        goal_values,
        home_expected_goals,
    )

    away_goal_probabilities = poisson.pmf(
        goal_values,
        away_expected_goals,
    )

    score_matrix = np.outer(
        home_goal_probabilities,
        away_goal_probabilities,
    )

    correction_factors = _dixon_coles_factors(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        rho=dixon_coles_rho,
    )

    for (
        home_goals,
        away_goals,
    ), factor in correction_factors.items():
        if factor <= 0:
            raise ValueError(
                "dixon_coles_rho creates an invalid "
                "non-positive score correction."
            )

        score_matrix[
            home_goals,
            away_goals,
        ] *= factor

    matrix_total = score_matrix.sum()

    if matrix_total <= 0:
        raise ValueError(
            "Score probabilities must have a positive total."
        )

    return score_matrix / matrix_total


def predict_match(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
    dixon_coles_rho: float = 0.0,
) -> dict:
    """
    Calculate match probabilities using a Poisson score model.

    When dixon_coles_rho is nonzero, low-scoring results receive
    the Dixon-Coles correction. A value of zero reproduces the
    independent Poisson model.
    """

    score_matrix = build_score_matrix(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        max_goals=max_goals,
        dixon_coles_rho=dixon_coles_rho,
    )

    goal_values = np.arange(max_goals + 1)

    home_win = np.tril(
        score_matrix,
        k=-1,
    ).sum()

    draw = np.trace(score_matrix)

    away_win = np.triu(
        score_matrix,
        k=1,
    ).sum()

    under_2_5 = sum(
        score_matrix[home_goals, away_goals]
        for home_goals in goal_values
        for away_goals in goal_values
        if home_goals + away_goals <= 2
    )

    over_2_5 = 1 - under_2_5

    both_teams_to_score = score_matrix[1:, 1:].sum()

    both_teams_not_to_score = (
        1 - both_teams_to_score
    )

    scorelines = []

    for home_goals in goal_values:
        for away_goals in goal_values:
            scorelines.append(
                {
                    "score": (
                        f"{home_goals}-{away_goals}"
                    ),
                    "probability": float(
                        score_matrix[
                            home_goals,
                            away_goals,
                        ]
                    ),
                }
            )

    top_scorelines = sorted(
        scorelines,
        key=lambda result: result["probability"],
        reverse=True,
    )[:5]

    display_max_goals = min(5, max_goals)

    display_score_matrix = [
        [
            float(
                score_matrix[
                    home_goals,
                    away_goals,
                ]
            )
            for away_goals in range(
                display_max_goals + 1
            )
        ]
        for home_goals in range(
            display_max_goals + 1
        )
    ]
    return {
        "model": {
            "name": (
                "dixon_coles"
                if dixon_coles_rho != 0
                else "independent_poisson"
            ),
            "dixon_coles_rho": float(
                dixon_coles_rho
            ),
        },
        "expected_goals": {
            "home": float(home_expected_goals),
            "away": float(away_expected_goals),
        },
        "score_matrix": {
            "max_goals": display_max_goals,
            "probabilities": display_score_matrix,
        },
        "match_result": {
            "home_win": float(home_win),
            "draw": float(draw),
            "away_win": float(away_win),
        },
        "total_goals": {
            "over_2_5": float(over_2_5),
            "under_2_5": float(under_2_5),
        },
        "both_teams_to_score": {
            "yes": float(both_teams_to_score),
            "no": float(both_teams_not_to_score),
        },
        "top_scorelines": top_scorelines,
    }
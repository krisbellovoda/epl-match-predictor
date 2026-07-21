import numpy as np
from scipy.stats import poisson


def predict_match(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
) -> dict:
    """
    Calculate match probabilities using independent Poisson distributions.

    Args:
        home_expected_goals: Expected goals for the home team.
        away_expected_goals: Expected goals for the away team.
        max_goals: Highest number of goals included for each team.

    Returns:
        A dictionary containing match and scoreline probabilities.
    """

    if home_expected_goals <= 0 or away_expected_goals <= 0:
        raise ValueError("Expected goals must be greater than zero.")

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

    # Normalize because extremely unlikely scores above max_goals
    # are excluded from the matrix.
    score_matrix = score_matrix / score_matrix.sum()

    home_win = np.tril(score_matrix, k=-1).sum()
    draw = np.trace(score_matrix)
    away_win = np.triu(score_matrix, k=1).sum()

    under_2_5 = sum(
        score_matrix[home_goals, away_goals]
        for home_goals in goal_values
        for away_goals in goal_values
        if home_goals + away_goals <= 2
    )

    over_2_5 = 1 - under_2_5

    both_teams_to_score = score_matrix[1:, 1:].sum()
    both_teams_not_to_score = 1 - both_teams_to_score

    scorelines = []

    for home_goals in goal_values:
        for away_goals in goal_values:
            scorelines.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": float(
                        score_matrix[home_goals, away_goals]
                    ),
                }
            )

    top_scorelines = sorted(
        scorelines,
        key=lambda result: result["probability"],
        reverse=True,
    )[:5]

    return {
        "expected_goals": {
            "home": home_expected_goals,
            "away": away_expected_goals,
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
import numpy as np
import pandas as pd

from backend.app.model.poisson import predict_match
from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


def run_backtest(
    matches: pd.DataFrame,
    minimum_training_matches: int = 100,
    half_life_days: float | None = None,
) -> dict:
    """
    Backtest the model chronologically.

    For every test match, only matches that happened earlier
    are used to calculate team strengths.
    """

    predictions = []

    for match_index in range(
        minimum_training_matches,
        len(matches),
    ):
        training_matches = matches.iloc[:match_index]
        test_match = matches.iloc[match_index]

        strength_model = build_team_strengths(
            training_matches,
            half_life_days=half_life_days,
        )

        try:
            home_xg, away_xg = estimate_expected_goals(
                home_team=test_match["HomeTeam"],
                away_team=test_match["AwayTeam"],
                strength_model=strength_model,
                allow_unknown=True,
            )
        except ValueError:
            continue

        prediction = predict_match(
            home_expected_goals=home_xg,
            away_expected_goals=away_xg,
        )

        result_probabilities = {
            "H": prediction["match_result"]["home_win"],
            "D": prediction["match_result"]["draw"],
            "A": prediction["match_result"]["away_win"],
        }

        predicted_result = max(
            result_probabilities,
            key=result_probabilities.get,
        )

        actual_result = test_match["FTR"]
        actual_probability = result_probabilities[
            actual_result
        ]

        actual_vector = {
            "H": np.array([1, 0, 0]),
            "D": np.array([0, 1, 0]),
            "A": np.array([0, 0, 1]),
        }[actual_result]

        predicted_vector = np.array(
            [
                result_probabilities["H"],
                result_probabilities["D"],
                result_probabilities["A"],
            ]
        )

        actual_total_goals = (
            test_match["FTHG"] + test_match["FTAG"]
        )

        predicted_over = (
            prediction["total_goals"]["over_2_5"]
            >= 0.5
        )

        actual_over = actual_total_goals >= 3

        predictions.append(
            {
                "date": test_match["Date"],
                "season": test_match["Season"],
                "home_team": test_match["HomeTeam"],
                "away_team": test_match["AwayTeam"],
                "actual_result": actual_result,
                "predicted_result": predicted_result,
                "correct_result": (
                    predicted_result == actual_result
                ),
                "actual_result_probability": (
                    actual_probability
                ),
                "brier_score": np.sum(
                    (
                        predicted_vector
                        - actual_vector
                    )
                    ** 2
                ),
                "over_2_5_probability":prediction[
                    "total_goals"
                ]["over_2_5"],
                "predicted_over_2_5": predicted_over,
                "actual_over_2_5": actual_over,
                "correct_over_2_5": (
                    predicted_over == actual_over
                ),
            }
        )
        
    results = pd.DataFrame(predictions)

    if results.empty:
        raise ValueError(
            "The backtest produced no predictions."
        )

    clipped_probabilities = np.clip(
        results["actual_result_probability"],
        1e-15,
        1,
    )

    summary = {
        "matches_tested": int(len(results)),
        "result_accuracy": float(
            results["correct_result"].mean()
        ),
        "over_2_5_accuracy": float(
            results["correct_over_2_5"].mean()
        ),
        "multiclass_brier_score": float(
            results["brier_score"].mean()
        ),
        "log_loss": float(
            -np.log(clipped_probabilities).mean()
        ),
    }

    return {
        "summary": summary,
        "predictions": results,
    }
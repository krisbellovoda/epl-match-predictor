import pandas as pd

from backend.app.model.hybrid_backtest import (
    run_hybrid_backtest,
)


def tune_baseline_model(
    matches: pd.DataFrame,
    validation_season: str = "2024_25",
    minimum_training_matches: int = 300,
) -> dict:
    """
    Tune team-strength settings using a validation season.

    adjustment_strength is fixed at zero, so recent-form
    adjustments are disabled. This evaluates only the original
    team-strength model.
    """

    prior_match_options = [
        3.0,
        5.0,
        10.0,
    ]

    half_life_options = [
        180.0,
        365.0,
        730.0,
        None,
    ]

    results = []

    total_combinations = (
        len(prior_match_options)
        * len(half_life_options)
    )

    combination_number = 0

    for prior_matches in prior_match_options:
        for half_life_days in half_life_options:
            combination_number += 1

            print(
                f"Testing {combination_number}/"
                f"{total_combinations}: "
                f"prior={prior_matches}, "
                f"half_life={half_life_days}"
            )

            backtest = run_hybrid_backtest(
                matches=matches,
                evaluation_season=validation_season,
                minimum_training_matches=(
                    minimum_training_matches
                ),
                prior_matches=prior_matches,
                half_life_days=half_life_days,
                adjustment_strength=0.0,
            )

            metrics = backtest["baseline"]

            results.append(
                {
                    "prior_matches": prior_matches,
                    "half_life_days": half_life_days,
                    "matches_tested": metrics[
                        "matches_tested"
                    ],
                    "result_accuracy": metrics[
                        "result_accuracy"
                    ],
                    "over_2_5_accuracy": metrics[
                        "over_2_5_accuracy"
                    ],
                    "multiclass_brier_score": metrics[
                        "multiclass_brier_score"
                    ],
                    "log_loss": metrics[
                        "log_loss"
                    ],
                    "total_goals_mae": metrics[
                        "total_goals_mae"
                    ],
                }
            )

    # Log loss is the primary selection metric.
    # Brier score breaks extremely close ties.
    ranked_results = sorted(
        results,
        key=lambda result: (
            result["log_loss"],
            result["multiclass_brier_score"],
        ),
    )

    return {
        "validation_season": validation_season,
        "selection_metric": "log_loss",
        "best_settings": ranked_results[0],
        "ranked_results": ranked_results,
    }
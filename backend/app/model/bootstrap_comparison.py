import numpy as np

from backend.app.model.poisson import predict_match


def _result_index(
    home_goals: int,
    away_goals: int,
) -> int:
    if home_goals > away_goals:
        return 0

    if home_goals == away_goals:
        return 1

    return 2


def _match_metric_differences(
    prediction_rows: list[dict],
) -> dict[str, np.ndarray]:
    """
    Calculate hybrid-minus-baseline metric differences for
    every individual match.

    For accuracy, positive values favor the hybrid.
    For Brier score and log loss, negative values favor it.
    """

    accuracy_differences = []
    brier_differences = []
    log_loss_differences = []

    for row in prediction_rows:
        score_parts = row["actual_score"].split("-")

        actual_home_goals = int(score_parts[0])
        actual_away_goals = int(score_parts[1])

        actual_result = _result_index(
            actual_home_goals,
            actual_away_goals,
        )

        baseline_prediction = predict_match(
            home_expected_goals=row[
                "baseline_home_xg"
            ],
            away_expected_goals=row[
                "baseline_away_xg"
            ],
        )

        hybrid_prediction = predict_match(
            home_expected_goals=row[
                "hybrid_home_xg"
            ],
            away_expected_goals=row[
                "hybrid_away_xg"
            ],
        )

        baseline_probabilities = np.array(
            [
                baseline_prediction["match_result"][
                    "home_win"
                ],
                baseline_prediction["match_result"][
                    "draw"
                ],
                baseline_prediction["match_result"][
                    "away_win"
                ],
            ],
            dtype=float,
        )

        hybrid_probabilities = np.array(
            [
                hybrid_prediction["match_result"][
                    "home_win"
                ],
                hybrid_prediction["match_result"][
                    "draw"
                ],
                hybrid_prediction["match_result"][
                    "away_win"
                ],
            ],
            dtype=float,
        )

        baseline_result = int(
            np.argmax(baseline_probabilities)
        )

        hybrid_result = int(
            np.argmax(hybrid_probabilities)
        )

        baseline_correct = float(
            baseline_result == actual_result
        )

        hybrid_correct = float(
            hybrid_result == actual_result
        )

        accuracy_differences.append(
            hybrid_correct - baseline_correct
        )

        actual_vector = np.zeros(3)
        actual_vector[actual_result] = 1.0

        baseline_brier = float(
            np.sum(
                (
                    baseline_probabilities
                    - actual_vector
                )
                ** 2
            )
        )

        hybrid_brier = float(
            np.sum(
                (
                    hybrid_probabilities
                    - actual_vector
                )
                ** 2
            )
        )

        brier_differences.append(
            hybrid_brier - baseline_brier
        )

        baseline_log_loss = float(
            -np.log(
                max(
                    baseline_probabilities[
                        actual_result
                    ],
                    1e-15,
                )
            )
        )

        hybrid_log_loss = float(
            -np.log(
                max(
                    hybrid_probabilities[
                        actual_result
                    ],
                    1e-15,
                )
            )
        )

        log_loss_differences.append(
            hybrid_log_loss - baseline_log_loss
        )

    return {
        "accuracy": np.array(
            accuracy_differences,
            dtype=float,
        ),
        "brier": np.array(
            brier_differences,
            dtype=float,
        ),
        "log_loss": np.array(
            log_loss_differences,
            dtype=float,
        ),
    }


def _sample_block_indices(
    number_of_matches: int,
    block_length: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """
    Generate a circular moving-block bootstrap sample.
    """

    sampled_indices = []

    while len(sampled_indices) < number_of_matches:
        block_start = int(
            random_generator.integers(
                0,
                number_of_matches,
            )
        )

        for offset in range(block_length):
            index = (
                block_start + offset
            ) % number_of_matches

            sampled_indices.append(index)

            if len(sampled_indices) == number_of_matches:
                break

    return np.array(
        sampled_indices,
        dtype=int,
    )


def _summarize_distribution(
    values: np.ndarray,
    lower_is_better: bool,
) -> dict:
    confidence_interval = np.percentile(
        values,
        [2.5, 97.5],
    )

    if lower_is_better:
        probability_hybrid_better = float(
            np.mean(values < 0)
        )
    else:
        probability_hybrid_better = float(
            np.mean(values > 0)
        )

    statistically_clear = not (
        confidence_interval[0] <= 0
        <= confidence_interval[1]
    )

    return {
        "mean_change": float(
            np.mean(values)
        ),
        "confidence_interval_95": {
            "lower": float(
                confidence_interval[0]
            ),
            "upper": float(
                confidence_interval[1]
            ),
        },
        "probability_hybrid_better": (
            probability_hybrid_better
        ),
        "interval_excludes_zero": bool(
            statistically_clear
        ),
        "better_direction": (
            "negative"
            if lower_is_better
            else "positive"
        ),
    }


def bootstrap_hybrid_comparison(
    backtest_result: dict,
    iterations: int = 5000,
    block_length: int = 10,
    random_seed: int = 42,
) -> dict:
    """
    Bootstrap the difference between baseline and hybrid metrics.

    Matches are sampled in chronological blocks to preserve some
    of the dependence between nearby fixtures.
    """

    if iterations < 100:
        raise ValueError(
            "iterations must be at least 100."
        )

    if block_length < 1:
        raise ValueError(
            "block_length must be at least one."
        )

    prediction_rows = backtest_result.get(
        "predictions",
        [],
    )

    if not prediction_rows:
        raise ValueError(
            "Backtest result contains no predictions."
        )

    differences = _match_metric_differences(
        prediction_rows
    )

    number_of_matches = len(prediction_rows)

    random_generator = np.random.default_rng(
        random_seed
    )

    bootstrap_accuracy = np.empty(iterations)
    bootstrap_brier = np.empty(iterations)
    bootstrap_log_loss = np.empty(iterations)

    for iteration in range(iterations):
        sampled_indices = _sample_block_indices(
            number_of_matches=number_of_matches,
            block_length=block_length,
            random_generator=random_generator,
        )

        bootstrap_accuracy[iteration] = np.mean(
            differences["accuracy"][
                sampled_indices
            ]
        )

        bootstrap_brier[iteration] = np.mean(
            differences["brier"][
                sampled_indices
            ]
        )

        bootstrap_log_loss[iteration] = np.mean(
            differences["log_loss"][
                sampled_indices
            ]
        )

    return {
        "matches": number_of_matches,
        "iterations": iterations,
        "block_length": block_length,
        "interpretation": (
            "Positive accuracy change favors the hybrid. "
            "Negative Brier and log-loss changes favor "
            "the hybrid."
        ),
        "accuracy": _summarize_distribution(
            bootstrap_accuracy,
            lower_is_better=False,
        ),
        "brier": _summarize_distribution(
            bootstrap_brier,
            lower_is_better=True,
        ),
        "log_loss": _summarize_distribution(
            bootstrap_log_loss,
            lower_is_better=True,
        ),
    }
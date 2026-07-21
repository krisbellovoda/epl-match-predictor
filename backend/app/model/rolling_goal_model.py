from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from backend.app.model.poisson import predict_match
from backend.app.model.rolling_features import (
    build_rolling_features,
)


BASE_FEATURES = [
    "home_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_for",
    "away_recent_goals_against",
    "home_recent_points",
    "away_recent_points",
    "home_venue_goals_for",
    "home_venue_goals_against",
    "away_venue_goals_for",
    "away_venue_goals_against",
]

SHOT_FEATURES = [
    "home_recent_shots",
    "away_recent_shots",
    "home_recent_shots_on_target",
    "away_recent_shots_on_target",
]


@dataclass
class PoissonRegressionModel:
    coefficients: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    feature_names: list[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate expected-goal estimates from match features.
        """

        feature_matrix = features[
            self.feature_names
        ].to_numpy(dtype=float)

        standardized = (
            feature_matrix - self.feature_means
        ) / self.feature_scales

        design_matrix = np.column_stack(
            [
                np.ones(len(standardized)),
                standardized,
            ]
        )

        linear_prediction = (
            design_matrix @ self.coefficients
        )

        # Avoid unreasonable expected-goal estimates.
        expected_goals = np.exp(
            np.clip(
                linear_prediction,
                np.log(0.05),
                np.log(6.0),
            )
        )

        return expected_goals


@dataclass
class RollingGoalModel:
    home_model: PoissonRegressionModel
    away_model: PoissonRegressionModel
    feature_names: list[str]

    def predict_expected_goals(
        self,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        home_expected_goals = self.home_model.predict(features)
        away_expected_goals = self.away_model.predict(features)

        return home_expected_goals, away_expected_goals


def _available_feature_names(
    matches: pd.DataFrame,
) -> list[str]:
    feature_names = BASE_FEATURES.copy()

    for feature_name in SHOT_FEATURES:
        if feature_name in matches.columns:
            feature_names.append(feature_name)

    return feature_names


def _poisson_negative_log_likelihood(
    coefficients: np.ndarray,
    design_matrix: np.ndarray,
    actual_goals: np.ndarray,
    ridge_strength: float,
) -> float:
    linear_prediction = design_matrix @ coefficients

    expected_goals = np.exp(
        np.clip(
            linear_prediction,
            -5.0,
            5.0,
        )
    )

    negative_log_likelihood = np.sum(
        expected_goals
        - actual_goals * linear_prediction
        + gammaln(actual_goals + 1)
    )

    # Do not penalize the intercept.
    ridge_penalty = ridge_strength * np.sum(
        coefficients[1:] ** 2
    )

    return float(
        negative_log_likelihood + ridge_penalty
    )


def _fit_single_poisson_model(
    matches: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    ridge_strength: float,
) -> PoissonRegressionModel:
    feature_matrix = matches[
        feature_names
    ].to_numpy(dtype=float)

    actual_goals = matches[
        target_column
    ].to_numpy(dtype=float)

    feature_means = feature_matrix.mean(axis=0)
    feature_scales = feature_matrix.std(axis=0)

    # A constant feature has a standard deviation of zero.
    feature_scales = np.where(
        feature_scales < 1e-8,
        1.0,
        feature_scales,
    )

    standardized = (
        feature_matrix - feature_means
    ) / feature_scales

    design_matrix = np.column_stack(
        [
            np.ones(len(standardized)),
            standardized,
        ]
    )

    initial_coefficients = np.zeros(
        design_matrix.shape[1]
    )

    initial_coefficients[0] = np.log(
        max(actual_goals.mean(), 0.05)
    )

    optimization = minimize(
        _poisson_negative_log_likelihood,
        initial_coefficients,
        args=(
            design_matrix,
            actual_goals,
            ridge_strength,
        ),
        method="L-BFGS-B",
    )

    if not optimization.success:
        raise RuntimeError(
            "Poisson regression fitting failed: "
            f"{optimization.message}"
        )

    return PoissonRegressionModel(
        coefficients=optimization.x,
        feature_means=feature_means,
        feature_scales=feature_scales,
        feature_names=feature_names,
    )


def fit_rolling_goal_model(
    training_matches: pd.DataFrame,
    ridge_strength: float = 1.0,
) -> RollingGoalModel:
    """
    Fit separate home-goal and away-goal Poisson regressions.
    """

    if training_matches.empty:
        raise ValueError("Training data cannot be empty.")

    feature_names = _available_feature_names(
        training_matches
    )

    missing_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in training_matches.columns
    ]

    if missing_features:
        raise ValueError(
            "Rolling features have not been created. "
            f"Missing: {', '.join(missing_features)}"
        )

    home_model = _fit_single_poisson_model(
        matches=training_matches,
        target_column="FTHG",
        feature_names=feature_names,
        ridge_strength=ridge_strength,
    )

    away_model = _fit_single_poisson_model(
        matches=training_matches,
        target_column="FTAG",
        feature_names=feature_names,
        ridge_strength=ridge_strength,
    )

    return RollingGoalModel(
        home_model=home_model,
        away_model=away_model,
        feature_names=feature_names,
    )


def _result_index(
    home_goals: int,
    away_goals: int,
) -> int:
    if home_goals > away_goals:
        return 0

    if home_goals == away_goals:
        return 1

    return 2


def evaluate_rolling_goal_model(
    matches: pd.DataFrame,
    evaluation_season: str = "2025_26",
    window: int = 5,
    minimum_history: int = 5,
    ridge_strength: float = 1.0,
) -> dict:
    """
    Train on seasons before evaluation_season and evaluate only
    on evaluation_season.
    """

    if "Season" not in matches.columns:
        raise ValueError(
            "Match data must contain a Season column."
        )

    featured_matches = build_rolling_features(
        matches,
        window=window,
    )

    usable_matches = featured_matches[
        (
            featured_matches["home_matches_available"]
            >= minimum_history
        )
        & (
            featured_matches["away_matches_available"]
            >= minimum_history
        )
    ].copy()

    training_matches = usable_matches[
        usable_matches["Season"] < evaluation_season
    ].copy()

    test_matches = usable_matches[
        usable_matches["Season"] == evaluation_season
    ].copy()

    if training_matches.empty:
        raise ValueError(
            "No earlier matches are available for training."
        )

    if test_matches.empty:
        raise ValueError(
            f"No matches found for season {evaluation_season}."
        )

    model = fit_rolling_goal_model(
        training_matches=training_matches,
        ridge_strength=ridge_strength,
    )

    home_xg_values, away_xg_values = (
        model.predict_expected_goals(test_matches)
    )

    correct_results = 0
    total_brier_score = 0.0
    total_log_loss = 0.0
    correct_over_2_5 = 0
    predictions = []

    for position, (_, match) in enumerate(
        test_matches.iterrows()
    ):
        home_xg = float(home_xg_values[position])
        away_xg = float(away_xg_values[position])

        prediction = predict_match(
            home_expected_goals=home_xg,
            away_expected_goals=away_xg,
        )

        probabilities = np.array(
            [
                prediction["match_result"]["home_win"],
                prediction["match_result"]["draw"],
                prediction["match_result"]["away_win"],
            ],
            dtype=float,
        )

        actual_result = _result_index(
            int(match["FTHG"]),
            int(match["FTAG"]),
        )

        predicted_result = int(
            np.argmax(probabilities)
        )

        if predicted_result == actual_result:
            correct_results += 1

        actual_vector = np.zeros(3)
        actual_vector[actual_result] = 1.0

        total_brier_score += float(
            np.sum(
                (probabilities - actual_vector) ** 2
            )
        )

        total_log_loss += float(
            -np.log(
                max(
                    probabilities[actual_result],
                    1e-15,
                )
            )
        )

        actual_over_2_5 = (
            int(match["FTHG"])
            + int(match["FTAG"])
            >= 3
        )

        predicted_over_2_5 = (
            prediction["total_goals"]["over_2_5"]
            >= 0.5
        )

        if predicted_over_2_5 == actual_over_2_5:
            correct_over_2_5 += 1

        predictions.append(
            {
                "date": match["Date"],
                "season": match["Season"],
                "home_team": match["HomeTeam"],
                "away_team": match["AwayTeam"],
                "actual_home_goals": int(match["FTHG"]),
                "actual_away_goals": int(match["FTAG"]),
                "home_expected_goals": home_xg,
                "away_expected_goals": away_xg,
                "home_win_probability": float(
                    probabilities[0]
                ),
                "draw_probability": float(
                    probabilities[1]
                ),
                "away_win_probability": float(
                    probabilities[2]
                ),
                "over_2_5_probability": prediction[
                    "total_goals"
                ]["over_2_5"],
            }
        )

    matches_tested = len(test_matches)

    return {
        "summary": {
            "model": "rolling-feature Poisson regression",
            "evaluation_season": evaluation_season,
            "training_matches": len(training_matches),
            "matches_tested": matches_tested,
            "window": window,
            "minimum_history": minimum_history,
            "features": model.feature_names,
            "result_accuracy": (
                correct_results / matches_tested
            ),
            "over_2_5_accuracy": (
                correct_over_2_5 / matches_tested
            ),
            "multiclass_brier_score": (
                total_brier_score / matches_tested
            ),
            "log_loss": (
                total_log_loss / matches_tested
            ),
            "average_home_expected_goals": float(
                np.mean(home_xg_values)
            ),
            "average_away_expected_goals": float(
                np.mean(away_xg_values)
            ),
        },
        "predictions": predictions,
    }
import numpy as np
import pandas as pd

from backend.app.model.poisson import predict_match
from backend.app.model.rolling_features import (
    build_rolling_features,
)
from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


def _actual_result_index(
    home_goals: int,
    away_goals: int,
) -> int:
    """
    Return 0 for home win, 1 for draw, and 2 for away win.
    """

    if home_goals > away_goals:
        return 0

    if home_goals == away_goals:
        return 1

    return 2


def _form_adjustment(
    recent_goals_for: float,
    opponent_recent_goals_against: float,
    league_team_goals: float,
    adjustment_strength: float,
    minimum_adjustment: float = 0.80,
    maximum_adjustment: float = 1.20,
) -> float:
    """
    Calculate a conservative expected-goals adjustment.

    A team scoring more than average and facing a defense
    conceding more than average receives an upward adjustment.
    """

    minimum_value = 0.05

    attacking_ratio = (
        max(recent_goals_for, minimum_value)
        / max(league_team_goals, minimum_value)
    )

    defensive_ratio = (
        max(opponent_recent_goals_against, minimum_value)
        / max(league_team_goals, minimum_value)
    )

    combined_ratio = np.sqrt(
        attacking_ratio * defensive_ratio
    )

    adjustment = np.exp(
        adjustment_strength
        * np.log(max(combined_ratio, minimum_value))
    )

    return float(
        np.clip(
            adjustment,
            minimum_adjustment,
            maximum_adjustment,
        )
    )


def calculate_hybrid_expected_goals(
    baseline_home_xg: float,
    baseline_away_xg: float,
    match_features: pd.Series,
    league_team_goals: float,
    adjustment_strength: float = 0.20,
) -> tuple[float, float]:
    """
    Blend long-term team strength with short-term form.

    adjustment_strength controls how much influence recent form
    receives. A value of zero returns the original baseline.
    """

    if not 0 <= adjustment_strength <= 1:
        raise ValueError(
            "adjustment_strength must be between zero and one."
        )

    home_adjustment = _form_adjustment(
        recent_goals_for=float(
            match_features["home_recent_goals_for"]
        ),
        opponent_recent_goals_against=float(
            match_features["away_recent_goals_against"]
        ),
        league_team_goals=league_team_goals,
        adjustment_strength=adjustment_strength,
    )

    away_adjustment = _form_adjustment(
        recent_goals_for=float(
            match_features["away_recent_goals_for"]
        ),
        opponent_recent_goals_against=float(
            match_features["home_recent_goals_against"]
        ),
        league_team_goals=league_team_goals,
        adjustment_strength=adjustment_strength,
    )

    hybrid_home_xg = np.clip(
        baseline_home_xg * home_adjustment,
        0.05,
        6.0,
    )

    hybrid_away_xg = np.clip(
        baseline_away_xg * away_adjustment,
        0.05,
        6.0,
    )

    return (
        float(hybrid_home_xg),
        float(hybrid_away_xg),
    )


def _empty_metrics() -> dict:
    return {
        "correct_results": 0,
        "correct_over_2_5": 0,
        "brier_total": 0.0,
        "log_loss_total": 0.0,
        "goal_error_total": 0.0,
    }


def _update_metrics(
    metrics: dict,
    prediction: dict,
    actual_home_goals: int,
    actual_away_goals: int,
) -> None:
    probabilities = np.array(
        [
            prediction["match_result"]["home_win"],
            prediction["match_result"]["draw"],
            prediction["match_result"]["away_win"],
        ],
        dtype=float,
    )

    actual_result = _actual_result_index(
        actual_home_goals,
        actual_away_goals,
    )

    predicted_result = int(
        np.argmax(probabilities)
    )

    if predicted_result == actual_result:
        metrics["correct_results"] += 1

    actual_vector = np.zeros(3)
    actual_vector[actual_result] = 1.0

    metrics["brier_total"] += float(
        np.sum(
            (probabilities - actual_vector) ** 2
        )
    )

    metrics["log_loss_total"] += float(
        -np.log(
            max(
                probabilities[actual_result],
                1e-15,
            )
        )
    )

    actual_over_2_5 = (
        actual_home_goals + actual_away_goals >= 3
    )

    predicted_over_2_5 = (
        prediction["total_goals"]["over_2_5"]
        >= 0.5
    )

    if predicted_over_2_5 == actual_over_2_5:
        metrics["correct_over_2_5"] += 1

    predicted_total_goals = (
        prediction["expected_goals"]["home"]
        + prediction["expected_goals"]["away"]
    )

    actual_total_goals = (
        actual_home_goals + actual_away_goals
    )

    metrics["goal_error_total"] += abs(
        predicted_total_goals - actual_total_goals
    )


def _summarize_metrics(
    metrics: dict,
    matches_tested: int,
) -> dict:
    return {
        "matches_tested": matches_tested,
        "result_accuracy": (
            metrics["correct_results"]
            / matches_tested
        ),
        "over_2_5_accuracy": (
            metrics["correct_over_2_5"]
            / matches_tested
        ),
        "multiclass_brier_score": (
            metrics["brier_total"]
            / matches_tested
        ),
        "log_loss": (
            metrics["log_loss_total"]
            / matches_tested
        ),
        "total_goals_mae": (
            metrics["goal_error_total"]
            / matches_tested
        ),
    }


def run_hybrid_backtest(
    matches: pd.DataFrame,
    evaluation_season: str = "2025_26",
    window: int = 5,
    minimum_training_matches: int = 760,
    prior_matches: float = 5.0,
    half_life_days: float | None = 365.0,
    adjustment_strength: float = 0.20,
) -> dict:
    """
    Compare the current team-strength baseline with a hybrid
    recent-form adjustment using chronological predictions.

    For each test match, the strength model is built using only
    matches that occurred before that match.
    """

    if "Season" not in matches.columns:
        raise ValueError(
            "Match data must contain a Season column."
        )

    featured_matches = build_rolling_features(
        matches,
        window=window,
    )

    evaluation_matches = featured_matches[
        featured_matches["Season"]
        == evaluation_season
    ].copy()

    if evaluation_matches.empty:
        raise ValueError(
            f"No matches found for season "
            f"{evaluation_season}."
        )

    baseline_metrics = _empty_metrics()
    hybrid_metrics = _empty_metrics()

    prediction_rows = []

    for _, test_match in evaluation_matches.iterrows():
        training_matches = featured_matches[
            featured_matches["Date"]
            < test_match["Date"]
        ].copy()

        if len(training_matches) < minimum_training_matches:
            continue

        home_team = test_match["HomeTeam"]
        away_team = test_match["AwayTeam"]

        strength_model = build_team_strengths(
            matches=training_matches,
            prior_matches=prior_matches,
            half_life_days=half_life_days,
        )

        baseline_home_xg, baseline_away_xg = (
            estimate_expected_goals(
                home_team=home_team,
                away_team=away_team,
                strength_model=strength_model,
                allow_unknown=True,
            )
        )

        league_team_goals = float(
            (
                training_matches["FTHG"].sum()
                + training_matches["FTAG"].sum()
            )
            / (2 * len(training_matches))
        )

        hybrid_home_xg, hybrid_away_xg = (
            calculate_hybrid_expected_goals(
                baseline_home_xg=baseline_home_xg,
                baseline_away_xg=baseline_away_xg,
                match_features=test_match,
                league_team_goals=league_team_goals,
                adjustment_strength=adjustment_strength,
            )
        )

        baseline_prediction = predict_match(
            home_expected_goals=baseline_home_xg,
            away_expected_goals=baseline_away_xg,
        )

        hybrid_prediction = predict_match(
            home_expected_goals=hybrid_home_xg,
            away_expected_goals=hybrid_away_xg,
        )

        actual_home_goals = int(
            test_match["FTHG"]
        )

        actual_away_goals = int(
            test_match["FTAG"]
        )

        _update_metrics(
            metrics=baseline_metrics,
            prediction=baseline_prediction,
            actual_home_goals=actual_home_goals,
            actual_away_goals=actual_away_goals,
        )

        _update_metrics(
            metrics=hybrid_metrics,
            prediction=hybrid_prediction,
            actual_home_goals=actual_home_goals,
            actual_away_goals=actual_away_goals,
        )

        prediction_rows.append(
            {
                "date": test_match["Date"],
                "home_team": home_team,
                "away_team": away_team,
                "actual_score": (
                    f"{actual_home_goals}-"
                    f"{actual_away_goals}"
                ),
                "baseline_home_xg": baseline_home_xg,
                "baseline_away_xg": baseline_away_xg,
                "hybrid_home_xg": hybrid_home_xg,
                "hybrid_away_xg": hybrid_away_xg,
            }
        )

    matches_tested = len(prediction_rows)

    if matches_tested == 0:
        raise ValueError(
            "No matches met the minimum training requirement."
        )

    baseline_summary = _summarize_metrics(
        baseline_metrics,
        matches_tested,
    )

    hybrid_summary = _summarize_metrics(
        hybrid_metrics,
        matches_tested,
    )

    comparison = {
        "result_accuracy_change": (
            hybrid_summary["result_accuracy"]
            - baseline_summary["result_accuracy"]
        ),
        "over_2_5_accuracy_change": (
            hybrid_summary["over_2_5_accuracy"]
            - baseline_summary["over_2_5_accuracy"]
        ),
        "brier_score_change": (
            hybrid_summary["multiclass_brier_score"]
            - baseline_summary["multiclass_brier_score"]
        ),
        "log_loss_change": (
            hybrid_summary["log_loss"]
            - baseline_summary["log_loss"]
        ),
        "total_goals_mae_change": (
            hybrid_summary["total_goals_mae"]
            - baseline_summary["total_goals_mae"]
        ),
    }

    return {
        "settings": {
            "evaluation_season": evaluation_season,
            "window": window,
            "minimum_training_matches": (
                minimum_training_matches
            ),
            "prior_matches": prior_matches,
            "half_life_days": half_life_days,
            "adjustment_strength": (
                adjustment_strength
            ),
        },
        "baseline": baseline_summary,
        "hybrid": hybrid_summary,
        "comparison": comparison,
        "predictions": prediction_rows,
    }
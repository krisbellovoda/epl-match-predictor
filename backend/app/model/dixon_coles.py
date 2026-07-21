import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


def _correction_factor(
    home_goals: int,
    away_goals: int,
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return (
            1
            - home_expected_goals
            * away_expected_goals
            * rho
        )

    if home_goals == 0 and away_goals == 1:
        return 1 + home_expected_goals * rho

    if home_goals == 1 and away_goals == 0:
        return 1 + away_expected_goals * rho

    if home_goals == 1 and away_goals == 1:
        return 1 - rho

    return 1.0


def _negative_log_likelihood(
    rho: float,
    observations: list[dict],
) -> float:
    total_log_likelihood = 0.0

    for observation in observations:
        home_goals = observation["home_goals"]
        away_goals = observation["away_goals"]
        home_xg = observation["home_expected_goals"]
        away_xg = observation["away_expected_goals"]

        correction = _correction_factor(
            home_goals=home_goals,
            away_goals=away_goals,
            home_expected_goals=home_xg,
            away_expected_goals=away_xg,
            rho=rho,
        )

        if correction <= 0:
            return float("inf")

        match_log_likelihood = (
            poisson.logpmf(home_goals, home_xg)
            + poisson.logpmf(away_goals, away_xg)
            + np.log(correction)
        )

        total_log_likelihood += match_log_likelihood

    return float(-total_log_likelihood)


def _safe_rho_bounds(
    observations: list[dict],
) -> tuple[float, float]:
    lower_bound = -0.25
    upper_bound = 0.25
    safety_margin = 1e-6

    for observation in observations:
        home_xg = observation["home_expected_goals"]
        away_xg = observation["away_expected_goals"]

        lower_bound = max(
            lower_bound,
            (-1 / home_xg) + safety_margin,
            (-1 / away_xg) + safety_margin,
        )

        upper_bound = min(
            upper_bound,
            (1 / (home_xg * away_xg))
            - safety_margin,
            1 - safety_margin,
        )

    if lower_bound >= upper_bound:
        raise ValueError(
            "Could not calculate valid rho bounds."
        )

    return lower_bound, upper_bound


def collect_rho_observations(
    matches: pd.DataFrame,
    fit_season: str = "2024_25",
    minimum_training_matches: int = 380,
    prior_matches: float = 5.0,
) -> list[dict]:
    """
    Generate expected-goal estimates chronologically.

    Every prediction uses only matches occurring before it.
    Only predictions from fit_season are retained for fitting rho.
    """

    required_columns = {
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    }

    missing_columns = required_columns.difference(
        matches.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    ordered_matches = matches.copy()

    ordered_matches["Date"] = pd.to_datetime(
        ordered_matches["Date"],
        dayfirst=True,
        errors="coerce",
    )

    ordered_matches = (
        ordered_matches.dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    observations = []

    for match_index in range(
        minimum_training_matches,
        len(ordered_matches),
    ):
        test_match = ordered_matches.iloc[match_index]

        if test_match["Season"] != fit_season:
            continue

        training_matches = ordered_matches.iloc[
            :match_index
        ]

        strength_model = build_team_strengths(
            training_matches,
            prior_matches=prior_matches,
        )

        home_xg, away_xg = estimate_expected_goals(
            home_team=test_match["HomeTeam"],
            away_team=test_match["AwayTeam"],
            strength_model=strength_model,
            allow_unknown=True,
        )

        observations.append(
            {
                "date": test_match["Date"],
                "season": test_match["Season"],
                "home_team": test_match["HomeTeam"],
                "away_team": test_match["AwayTeam"],
                "home_goals": int(test_match["FTHG"]),
                "away_goals": int(test_match["FTAG"]),
                "home_expected_goals": float(home_xg),
                "away_expected_goals": float(away_xg),
            }
        )

    if not observations:
        raise ValueError(
            f"No observations found for {fit_season}."
        )

    return observations


def fit_dixon_coles_rho(
    matches: pd.DataFrame,
    fit_season: str = "2024_25",
    minimum_training_matches: int = 380,
    prior_matches: float = 5.0,
) -> dict:
    observations = collect_rho_observations(
        matches=matches,
        fit_season=fit_season,
        minimum_training_matches=(
            minimum_training_matches
        ),
        prior_matches=prior_matches,
    )

    lower_bound, upper_bound = _safe_rho_bounds(
        observations
    )

    optimization_result = minimize_scalar(
        lambda rho: _negative_log_likelihood(
            rho,
            observations,
        ),
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={
            "xatol": 1e-8,
        },
    )

    if not optimization_result.success:
        raise RuntimeError(
            "Dixon-Coles rho optimization failed."
        )

    fitted_rho = float(optimization_result.x)

    baseline_negative_log_likelihood = (
        _negative_log_likelihood(
            0.0,
            observations,
        )
    )

    fitted_negative_log_likelihood = (
        _negative_log_likelihood(
            fitted_rho,
            observations,
        )
    )

    return {
        "fit_season": fit_season,
        "matches": len(observations),
        "rho": fitted_rho,
        "baseline_negative_log_likelihood": (
            baseline_negative_log_likelihood
        ),
        "fitted_negative_log_likelihood": (
            fitted_negative_log_likelihood
        ),
        "improvement": (
            baseline_negative_log_likelihood
            - fitted_negative_log_likelihood
        ),
        "bounds": {
            "lower": float(lower_bound),
            "upper": float(upper_bound),
        },
    }
def evaluate_dixon_coles_rho(
    matches: pd.DataFrame,
    rho: float,
    evaluation_season: str = "2025_26",
    minimum_training_matches: int = 760,
    prior_matches: float = 5.0,
) -> dict:
    """
    Compare independent Poisson and a fixed Dixon-Coles rho
    on a later chronological evaluation season.
    """

    observations = collect_rho_observations(
        matches=matches,
        fit_season=evaluation_season,
        minimum_training_matches=(
            minimum_training_matches
        ),
        prior_matches=prior_matches,
    )

    baseline_nll = _negative_log_likelihood(
        0.0,
        observations,
    )

    adjusted_nll = _negative_log_likelihood(
        rho,
        observations,
    )

    matches_tested = len(observations)

    return {
        "evaluation_season": evaluation_season,
        "matches": matches_tested,
        "rho": float(rho),
        "baseline_negative_log_likelihood": (
            baseline_nll
        ),
        "adjusted_negative_log_likelihood": (
            adjusted_nll
        ),
        "total_improvement": (
            baseline_nll - adjusted_nll
        ),
        "average_baseline_log_loss": (
            baseline_nll / matches_tested
        ),
        "average_adjusted_log_loss": (
            adjusted_nll / matches_tested
        ),
    }
import numpy as np
import pandas as pd


def build_team_strengths(
    matches: pd.DataFrame,
    prior_matches: float = 5.0,
    half_life_days: float | None = None,
) -> dict:
    """
    Calculate attacking and defensive team strengths.

    Small samples are shrunk toward league-average performance.
    When half_life_days is provided, recent matches receive more
    weight than older matches.
    """

    if matches.empty:
        raise ValueError(
            "Cannot build team strengths from empty match data."
        )

    if prior_matches <= 0:
        raise ValueError(
            "prior_matches must be greater than zero."
        )

    if (
        half_life_days is not None
        and half_life_days <= 0
    ):
        raise ValueError(
            "half_life_days must be greater than zero."
        )

    weighted_matches = matches.copy()

    if half_life_days is None:
        weighted_matches["weight"] = 1.0
    else:
        most_recent_date = weighted_matches["Date"].max()

        age_in_days = (
            most_recent_date - weighted_matches["Date"]
        ).dt.days.clip(lower=0)

        weighted_matches["weight"] = np.power(
            0.5,
            age_in_days / half_life_days,
        )

    weighted_matches["weighted_home_goals"] = (
        weighted_matches["FTHG"]
        * weighted_matches["weight"]
    )

    weighted_matches["weighted_away_goals"] = (
        weighted_matches["FTAG"]
        * weighted_matches["weight"]
    )

    total_weight = weighted_matches["weight"].sum()

    league_home_goals = (
        weighted_matches["weighted_home_goals"].sum()
        / total_weight
    )

    league_away_goals = (
        weighted_matches["weighted_away_goals"].sum()
        / total_weight
    )

    home_statistics = weighted_matches.groupby(
        "HomeTeam"
    ).agg(
        home_goals_scored=(
            "weighted_home_goals",
            "sum",
        ),
        home_goals_conceded=(
            "weighted_away_goals",
            "sum",
        ),
        home_weight=("weight", "sum"),
    )

    away_statistics = weighted_matches.groupby(
        "AwayTeam"
    ).agg(
        away_goals_scored=(
            "weighted_away_goals",
            "sum",
        ),
        away_goals_conceded=(
            "weighted_home_goals",
            "sum",
        ),
        away_weight=("weight", "sum"),
    )

    team_statistics = home_statistics.join(
        away_statistics,
        how="outer",
    ).fillna(0)

    team_statistics["smoothed_home_scored"] = (
        team_statistics["home_goals_scored"]
        + prior_matches * league_home_goals
    ) / (
        team_statistics["home_weight"]
        + prior_matches
    )

    team_statistics["smoothed_home_conceded"] = (
        team_statistics["home_goals_conceded"]
        + prior_matches * league_away_goals
    ) / (
        team_statistics["home_weight"]
        + prior_matches
    )

    team_statistics["smoothed_away_scored"] = (
        team_statistics["away_goals_scored"]
        + prior_matches * league_away_goals
    ) / (
        team_statistics["away_weight"]
        + prior_matches
    )

    team_statistics["smoothed_away_conceded"] = (
        team_statistics["away_goals_conceded"]
        + prior_matches * league_home_goals
    ) / (
        team_statistics["away_weight"]
        + prior_matches
    )

    team_statistics["home_attack"] = (
        team_statistics["smoothed_home_scored"]
        / league_home_goals
    )

    team_statistics["home_defense"] = (
        team_statistics["smoothed_home_conceded"]
        / league_away_goals
    )

    team_statistics["away_attack"] = (
        team_statistics["smoothed_away_scored"]
        / league_away_goals
    )

    team_statistics["away_defense"] = (
        team_statistics["smoothed_away_conceded"]
        / league_home_goals
    )

    return {
        "league_home_goals": float(
            league_home_goals
        ),
        "league_away_goals": float(
            league_away_goals
        ),
        "prior_matches": float(prior_matches),
        "half_life_days": half_life_days,
        "teams": team_statistics,
    }


def estimate_expected_goals(
    home_team: str,
    away_team: str,
    strength_model: dict,
    allow_unknown: bool = False,
) -> tuple[float, float]:
    """
    Estimate expected goals for a home-versus-away matchup.
    """

    if home_team == away_team:
        raise ValueError(
            "Home and away teams must be different."
        )

    teams = strength_model["teams"]

    neutral_strength = pd.Series(
        {
            "home_attack": 1.0,
            "home_defense": 1.0,
            "away_attack": 1.0,
            "away_defense": 1.0,
        }
    )

    if home_team in teams.index:
        home_strength = teams.loc[home_team]
    elif allow_unknown:
        home_strength = neutral_strength
    else:
        raise ValueError(
            f"Unknown home team: {home_team}"
        )

    if away_team in teams.index:
        away_strength = teams.loc[away_team]
    elif allow_unknown:
        away_strength = neutral_strength
    else:
        raise ValueError(
            f"Unknown away team: {away_team}"
        )

    home_expected_goals = (
        strength_model["league_home_goals"]
        * home_strength["home_attack"]
        * away_strength["away_defense"]
    )

    away_expected_goals = (
        strength_model["league_away_goals"]
        * away_strength["away_attack"]
        * home_strength["home_defense"]
    )

    return (
        float(home_expected_goals),
        float(away_expected_goals),
    )
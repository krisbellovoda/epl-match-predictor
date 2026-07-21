from collections import defaultdict, deque

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
}


def _safe_mean(
    values: deque,
    fallback: float,
) -> float:
    """
    Return the mean of previous values.

    The fallback is used when no historical values exist.
    """

    if not values:
        return float(fallback)

    return float(np.mean(values))


def _safe_number(
    value,
    fallback: float,
) -> float:
    """
    Convert a value to float while handling missing data.
    """

    if pd.isna(value):
        return float(fallback)

    return float(value)


def _validate_matches(matches: pd.DataFrame) -> None:
    """
    Check that the match data contains the required information.
    """

    if matches.empty:
        raise ValueError("Match data cannot be empty.")

    missing_columns = REQUIRED_COLUMNS - set(matches.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Match data is missing required columns: "
            f"{missing_text}"
        )

    if matches["HomeTeam"].isna().any():
        raise ValueError(
            "One or more matches are missing a home team."
        )

    if matches["AwayTeam"].isna().any():
        raise ValueError(
            "One or more matches are missing an away team."
        )

    if matches["FTHG"].isna().any():
        raise ValueError(
            "One or more matches are missing home goals."
        )

    if matches["FTAG"].isna().any():
        raise ValueError(
            "One or more matches are missing away goals."
        )


def _prepare_matches(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a chronologically sorted copy of the match data.
    """

    prepared = matches.copy()

    prepared["Date"] = pd.to_datetime(
        prepared["Date"],
        dayfirst=True,
        errors="coerce",
    )

    if prepared["Date"].isna().any():
        raise ValueError(
            "One or more match dates could not be parsed."
        )

    sort_columns = ["Date"]

    if "Season" in prepared.columns:
        sort_columns = ["Season", "Date"]

    return prepared.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)


def build_rolling_features(
    matches: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Create leakage-safe rolling pre-match features.

    Every feature for a match is calculated before that match is
    added to team and league history. The current match therefore
    cannot affect its own features.

    Args:
        matches:
            Historical match data.

        window:
            Maximum number of previous team matches used for
            recent-form features.

    Returns:
        Match data with additional pre-match feature columns.
    """

    if window < 1:
        raise ValueError(
            "Rolling window must be at least one."
        )

    _validate_matches(matches)

    prepared = _prepare_matches(matches)

    # These small prior samples provide reasonable fallbacks for
    # the first matches in the dataset. They are not calculated
    # from future match data.
    league_home_goals = [1.5]
    league_away_goals = [1.2]
    league_team_goals = [1.5, 1.2]

    league_home_shots = [13.0]
    league_away_shots = [11.0]

    league_home_shots_on_target = [4.5]
    league_away_shots_on_target = [3.8]

    # Each team receives its own collection of rolling histories.
    histories = defaultdict(
        lambda: {
            "goals_for": deque(maxlen=window),
            "goals_against": deque(maxlen=window),
            "points": deque(maxlen=window),
            "shots": deque(maxlen=window),
            "shots_on_target": deque(maxlen=window),
            "home_goals_for": deque(maxlen=window),
            "home_goals_against": deque(maxlen=window),
            "away_goals_for": deque(maxlen=window),
            "away_goals_against": deque(maxlen=window),
        }
    )

    feature_rows = []

    for _, match in prepared.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_history = histories[home_team]
        away_history = histories[away_team]

        # These expanding league averages contain only matches
        # that occurred before the current match.
        league_home_goal_average = float(
            np.mean(league_home_goals)
        )

        league_away_goal_average = float(
            np.mean(league_away_goals)
        )

        league_team_goal_average = float(
            np.mean(league_team_goals)
        )

        home_shots_fallback = float(
            np.mean(league_home_shots)
        )

        away_shots_fallback = float(
            np.mean(league_away_shots)
        )

        home_sot_fallback = float(
            np.mean(league_home_shots_on_target)
        )

        away_sot_fallback = float(
            np.mean(league_away_shots_on_target)
        )

        # Everything in this dictionary is known before kickoff.
        features = {
            "home_recent_goals_for": _safe_mean(
                home_history["goals_for"],
                league_team_goal_average,
            ),
            "home_recent_goals_against": _safe_mean(
                home_history["goals_against"],
                league_team_goal_average,
            ),
            "away_recent_goals_for": _safe_mean(
                away_history["goals_for"],
                league_team_goal_average,
            ),
            "away_recent_goals_against": _safe_mean(
                away_history["goals_against"],
                league_team_goal_average,
            ),
            "home_recent_points": _safe_mean(
                home_history["points"],
                1.0,
            ),
            "away_recent_points": _safe_mean(
                away_history["points"],
                1.0,
            ),
            "home_venue_goals_for": _safe_mean(
                home_history["home_goals_for"],
                league_home_goal_average,
            ),
            "home_venue_goals_against": _safe_mean(
                home_history["home_goals_against"],
                league_away_goal_average,
            ),
            "away_venue_goals_for": _safe_mean(
                away_history["away_goals_for"],
                league_away_goal_average,
            ),
            "away_venue_goals_against": _safe_mean(
                away_history["away_goals_against"],
                league_home_goal_average,
            ),
            "home_matches_available": len(
                home_history["goals_for"]
            ),
            "away_matches_available": len(
                away_history["goals_for"]
            ),
        }

        has_shots = (
            "HS" in prepared.columns
            and "AS" in prepared.columns
        )

        has_shots_on_target = (
            "HST" in prepared.columns
            and "AST" in prepared.columns
        )

        if has_shots:
            features.update(
                {
                    "home_recent_shots": _safe_mean(
                        home_history["shots"],
                        home_shots_fallback,
                    ),
                    "away_recent_shots": _safe_mean(
                        away_history["shots"],
                        away_shots_fallback,
                    ),
                }
            )

        if has_shots_on_target:
            features.update(
                {
                    "home_recent_shots_on_target": _safe_mean(
                        home_history["shots_on_target"],
                        home_sot_fallback,
                    ),
                    "away_recent_shots_on_target": _safe_mean(
                        away_history["shots_on_target"],
                        away_sot_fallback,
                    ),
                }
            )

        feature_rows.append(features)

        # -----------------------------------------------------
        # The match has now "happened."
        #
        # Only after its pre-match features were recorded do we
        # add its result and statistics to historical data.
        # -----------------------------------------------------

        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])

        if home_goals > away_goals:
            home_points = 3.0
            away_points = 0.0
        elif home_goals < away_goals:
            home_points = 0.0
            away_points = 3.0
        else:
            home_points = 1.0
            away_points = 1.0

        # Update the home team's overall history.
        home_history["goals_for"].append(
            home_goals
        )
        home_history["goals_against"].append(
            away_goals
        )
        home_history["points"].append(
            home_points
        )

        # Update the home team's home-only history.
        home_history["home_goals_for"].append(
            home_goals
        )
        home_history["home_goals_against"].append(
            away_goals
        )

        # Update the away team's overall history.
        away_history["goals_for"].append(
            away_goals
        )
        away_history["goals_against"].append(
            home_goals
        )
        away_history["points"].append(
            away_points
        )

        # Update the away team's away-only history.
        away_history["away_goals_for"].append(
            away_goals
        )
        away_history["away_goals_against"].append(
            home_goals
        )

        if has_shots:
            home_shots = _safe_number(
                match["HS"],
                home_shots_fallback,
            )

            away_shots = _safe_number(
                match["AS"],
                away_shots_fallback,
            )

            home_history["shots"].append(
                home_shots
            )

            away_history["shots"].append(
                away_shots
            )

            league_home_shots.append(
                home_shots
            )

            league_away_shots.append(
                away_shots
            )

        if has_shots_on_target:
            home_shots_on_target = _safe_number(
                match["HST"],
                home_sot_fallback,
            )

            away_shots_on_target = _safe_number(
                match["AST"],
                away_sot_fallback,
            )

            home_history["shots_on_target"].append(
                home_shots_on_target
            )

            away_history["shots_on_target"].append(
                away_shots_on_target
            )

            league_home_shots_on_target.append(
                home_shots_on_target
            )

            league_away_shots_on_target.append(
                away_shots_on_target
            )

        # Update expanding league goal histories last.
        league_home_goals.append(
            home_goals
        )

        league_away_goals.append(
            away_goals
        )

        league_team_goals.extend(
            [
                home_goals,
                away_goals,
            ]
        )

    feature_frame = pd.DataFrame(
        feature_rows
    )

    return pd.concat(
        [
            prepared.reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )
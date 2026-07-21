import numpy as np
import pandas as pd

from backend.app.model.calibration_service import OVER_CALIBRATOR
from backend.app.model.market import compare_two_way_market
from backend.app.model.poisson import predict_match
from backend.app.model.probability_calibrator import (
    apply_probability_calibration,
)
from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


def run_betting_backtest(
    matches: pd.DataFrame,
    minimum_training_matches: int = 760,
    minimum_edge: float = 0.05,
    prior_matches: float = 5.0,
    test_season: str | None = None,
) -> dict:
    """
    Backtest flat one-unit Over/Under 2.5 bets.

    For every tested match:
    1. Train the strength model using earlier matches only.
    2. Generate the raw Poisson probability.
    3. Apply probability calibration.
    4. Compare the calibrated probability with no-vig market odds.
    5. Place a bet only when edge and expected value are sufficient.

    When test_season is provided, only matches from that season are
    evaluated. Earlier seasons remain available as training data.
    """

    if not 0 <= minimum_edge <= 1:
        raise ValueError(
            "minimum_edge must be between 0 and 1."
        )

    if minimum_training_matches < 1:
        raise ValueError(
            "minimum_training_matches must be positive."
        )

    required_columns = {
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "Avg>2.5",
        "Avg<2.5",
    }

    missing_columns = required_columns.difference(
        matches.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Make chronological ordering explicit.
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(
        matches["Date"],
        dayfirst=True,
        errors="coerce",
    )

    matches = (
        matches.dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    bets = []

    for match_index in range(
        minimum_training_matches,
        len(matches),
    ):
        training_matches = matches.iloc[:match_index]
        test_match = matches.iloc[match_index]

        # Skip matches outside the requested holdout season.
        if (
            test_season is not None
            and test_match["Season"] != test_season
        ):
            continue

        over_odds = test_match["Avg>2.5"]
        under_odds = test_match["Avg<2.5"]

        if (
            pd.isna(over_odds)
            or pd.isna(under_odds)
        ):
            continue

        over_odds = float(over_odds)
        under_odds = float(under_odds)

        if over_odds <= 1 or under_odds <= 1:
            continue

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

        prediction = predict_match(
            home_expected_goals=home_xg,
            away_expected_goals=away_xg,
        )

        raw_over_probability = float(
            prediction["total_goals"]["over_2_5"]
        )

        calibrated_over_probability = float(
            apply_probability_calibration(
                [raw_over_probability],
                OVER_CALIBRATOR,
            )[0]
        )

        market_comparison = compare_two_way_market(
            model_over_probability=(
                calibrated_over_probability
            ),
            over_decimal_odds=over_odds,
            under_decimal_odds=under_odds,
        )

        over_edge = float(
            market_comparison["edge"]["over_2_5"]
        )

        under_edge = float(
            market_comparison["edge"]["under_2_5"]
        )

        over_value = float(
            market_comparison[
                "expected_value"
            ]["over_2_5"]
        )

        under_value = float(
            market_comparison[
                "expected_value"
            ]["under_2_5"]
        )

        if over_edge >= under_edge:
            selected_market = "over_2_5"
            selected_edge = over_edge
            selected_value = over_value
            selected_odds = over_odds
        else:
            selected_market = "under_2_5"
            selected_edge = under_edge
            selected_value = under_value
            selected_odds = under_odds

        if selected_edge < minimum_edge:
            continue

        if selected_value <= 0:
            continue

        total_goals = (
            int(test_match["FTHG"])
            + int(test_match["FTAG"])
        )

        actual_over = total_goals >= 3

        if selected_market == "over_2_5":
            bet_won = actual_over
        else:
            bet_won = not actual_over

        if bet_won:
            profit = selected_odds - 1
        else:
            profit = -1.0

        bets.append(
            {
                "date": test_match["Date"],
                "season": test_match["Season"],
                "home_team": test_match["HomeTeam"],
                "away_team": test_match["AwayTeam"],
                "selected_market": selected_market,
                "raw_over_probability": (
                    raw_over_probability
                ),
                "calibrated_over_probability": (
                    calibrated_over_probability
                ),
                "model_edge": selected_edge,
                "estimated_value": selected_value,
                "decimal_odds": selected_odds,
                "actual_total_goals": total_goals,
                "bet_won": bool(bet_won),
                "profit": float(profit),
            }
        )

    bet_results = pd.DataFrame(bets)

    if bet_results.empty:
        return {
            "summary": {
                "test_season": test_season,
                "minimum_edge": float(minimum_edge),
                "bets": 0,
                "wins": 0,
                "losses": 0,
                "hit_rate": 0.0,
                "profit_units": 0.0,
                "roi": 0.0,
                "average_odds": 0.0,
                "average_edge": 0.0,
                "maximum_drawdown": 0.0,
            },
            "bets": bet_results,
        }

    bet_results["cumulative_profit"] = (
        bet_results["profit"].cumsum()
    )

    running_peak = np.maximum.accumulate(
        np.maximum(
            bet_results["cumulative_profit"],
            0.0,
        )
    )

    drawdown = (
        bet_results["cumulative_profit"]
        - running_peak
    )

    total_profit = float(
        bet_results["profit"].sum()
    )

    total_staked = len(bet_results)
    wins = int(bet_results["bet_won"].sum())

    summary = {
        "test_season": test_season,
        "minimum_edge": float(minimum_edge),
        "bets": int(total_staked),
        "wins": wins,
        "losses": int(total_staked - wins),
        "hit_rate": float(
            bet_results["bet_won"].mean()
        ),
        "profit_units": total_profit,
        "roi": float(
            total_profit / total_staked
        ),
        "average_odds": float(
            bet_results["decimal_odds"].mean()
        ),
        "average_edge": float(
            bet_results["model_edge"].mean()
        ),
        "maximum_drawdown": float(
            abs(drawdown.min())
        ),
    }

    return {
        "summary": summary,
        "bets": bet_results,
    }
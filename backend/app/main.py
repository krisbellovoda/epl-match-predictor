import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.model.calibration_service import (
    calibrate_prediction_totals,
)
from backend.app.model.config import (
    CURRENT_EPL_TEAMS,
    CURRENT_SEASON,
    MODEL_VERSION,
    get_model_configuration,
)
from backend.app.model.market import (
    compare_two_way_market,
)
from backend.app.model.model_artifact import (
    load_production_model,
)
from backend.app.model.poisson import predict_match
from backend.app.model.team_strength import (
    estimate_expected_goals,
)


app = FastAPI(
    title="EPL Match Prediction API",
    description=(
        "API for an educational EPL football "
        "probability model"
    ),
    version=MODEL_VERSION,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

frontend_origins_text = os.getenv(
    "FRONTEND_ORIGINS",
    DEFAULT_FRONTEND_ORIGINS,
)

allowed_frontend_origins = [
    origin.strip().rstrip("/")
    for origin in frontend_origins_text.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------

BACKTEST_SUMMARY_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "artifacts"
    / "backtest_summary.json"
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    home_expected_goals: float = Field(
        gt=0,
        le=10,
        examples=[1.75],
    )

    away_expected_goals: float = Field(
        gt=0,
        le=10,
        examples=[0.90],
    )


class TeamPredictionRequest(BaseModel):
    home_team: str = Field(
        min_length=1,
        examples=["Arsenal"],
    )

    away_team: str = Field(
        min_length=1,
        examples=["Chelsea"],
    )


class MarketComparisonRequest(BaseModel):
    model_over_probability: float = Field(
        ge=0,
        le=1,
        examples=[0.55],
    )

    over_decimal_odds: float = Field(
        gt=1,
        examples=[1.95],
    )

    under_decimal_odds: float = Field(
        gt=1,
        examples=[1.90],
    )


# ---------------------------------------------------------
# Load production model
# ---------------------------------------------------------

# The deployed API loads the versioned production artifact.
# Raw CSV files are only needed when retraining the model.

strength_model, model_metadata = (
    load_production_model()
)


def validate_production_roster() -> None:
    """
    Confirm that the production artifact contains every club
    in the configured 2026-27 EPL roster.
    """

    loaded_teams = set(
        strength_model["teams"].index.tolist()
    )

    expected_teams = set(CURRENT_EPL_TEAMS)

    missing_teams = sorted(
        expected_teams - loaded_teams
    )

    unexpected_teams = sorted(
        loaded_teams - expected_teams
    )

    if missing_teams:
        raise RuntimeError(
            "Production model is missing current EPL teams: "
            + ", ".join(missing_teams)
        )

    if unexpected_teams:
        raise RuntimeError(
            "Production model contains teams outside the "
            "current EPL roster: "
            + ", ".join(unexpected_teams)
        )


validate_production_roster()


# ---------------------------------------------------------
# General endpoints
# ---------------------------------------------------------

@app.get(
    "/",
    summary="API Home",
)
def home():
    return {
        "message": (
            "EPL Match Prediction API is running"
        ),
        "model_version": MODEL_VERSION,
        "season": CURRENT_SEASON,
        "documentation": "/docs",
    }


@app.get(
    "/health",
    summary="Health Check",
)
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "season": CURRENT_SEASON,
        "training_matches": model_metadata[
            "training_matches"
        ],
        "teams_loaded": len(
            strength_model["teams"]
        ),
        "selectable_teams": len(
            CURRENT_EPL_TEAMS
        ),
        "prior_based_teams": model_metadata.get(
            "prior_based_teams",
            [],
        ),
        "artifact": {
            "generated_at": model_metadata[
                "generated_at"
            ],
            "last_training_match": model_metadata[
                "last_training_match"
            ],
            "schema_version": model_metadata[
                "schema_version"
            ],
        },
        "model": {
            "prior_matches": (
                strength_model["prior_matches"]
            ),
            "half_life_days": (
                strength_model["half_life_days"]
            ),
        },
    }


@app.get(
    "/model/info",
    summary="Production Model Information",
)
def model_info():
    configuration = get_model_configuration()

    return {
        **configuration,
        "artifact": {
            "generated_at": model_metadata[
                "generated_at"
            ],
            "training_matches": model_metadata[
                "training_matches"
            ],
            "last_training_match": model_metadata[
                "last_training_match"
            ],
            "schema_version": model_metadata[
                "schema_version"
            ],
            "season": model_metadata.get(
                "season",
                CURRENT_SEASON,
            ),
            "prior_based_teams": (
                model_metadata.get(
                    "prior_based_teams",
                    [],
                )
            ),
        },
    }


@app.get(
    "/model/performance",
    summary="Model Performance",
)
def get_model_performance():
    if not BACKTEST_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model performance summary is not "
                "currently available."
            ),
        )

    try:
        with BACKTEST_SUMMARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as summary_file:
            return json.load(summary_file)

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model performance summary contains "
                "invalid JSON."
            ),
        ) from error


@app.get(
    "/teams",
    summary="Available 2026-27 EPL Teams",
)
def get_teams():
    return {
        "teams": list(CURRENT_EPL_TEAMS),
        "count": len(CURRENT_EPL_TEAMS),
        "season": CURRENT_SEASON,
        "model_version": MODEL_VERSION,
        "prior_based_teams": model_metadata.get(
            "prior_based_teams",
            [],
        ),
    }


# ---------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------

@app.post(
    "/predict",
    summary="Manual Expected-Goals Prediction",
)
def create_manual_prediction(
    request: PredictionRequest,
):
    prediction = predict_match(
        home_expected_goals=(
            request.home_expected_goals
        ),
        away_expected_goals=(
            request.away_expected_goals
        ),
    )

    calibrated_prediction = (
        calibrate_prediction_totals(
            prediction
        )
    )

    return {
        "model_version": MODEL_VERSION,
        "season": CURRENT_SEASON,
        **calibrated_prediction,
    }


@app.post(
    "/predict/teams",
    summary="2026-27 EPL Team Match Prediction",
)
def create_team_prediction(
    request: TeamPredictionRequest,
):
    if request.home_team == request.away_team:
        raise HTTPException(
            status_code=400,
            detail=(
                "Home and away teams must be different."
            ),
        )

    if request.home_team not in CURRENT_EPL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{request.home_team} is not a selectable "
                f"{CURRENT_SEASON} EPL team."
            ),
        )

    if request.away_team not in CURRENT_EPL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{request.away_team} is not a selectable "
                f"{CURRENT_SEASON} EPL team."
            ),
        )

    try:
        home_xg, away_xg = estimate_expected_goals(
            home_team=request.home_team,
            away_team=request.away_team,
            strength_model=strength_model,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    prediction = predict_match(
        home_expected_goals=home_xg,
        away_expected_goals=away_xg,
    )

    calibrated_prediction = (
        calibrate_prediction_totals(
            prediction
        )
    )

    prior_based_teams = set(
        model_metadata.get(
            "prior_based_teams",
            [],
        )
    )

    prediction_uses_prior = (
        request.home_team in prior_based_teams
        or request.away_team in prior_based_teams
    )

    return {
        "model_version": MODEL_VERSION,
        "season": CURRENT_SEASON,
        "home_team": request.home_team,
        "away_team": request.away_team,
        "prediction_uses_promoted_team_prior": (
            prediction_uses_prior
        ),
        **calibrated_prediction,
    }


# ---------------------------------------------------------
# Sportsbook comparison
# ---------------------------------------------------------

@app.post(
    "/market/compare",
    summary="Compare Model With Sportsbook Market",
)
def compare_market(
    request: MarketComparisonRequest,
):
    try:
        comparison = compare_two_way_market(
            model_over_probability=(
                request.model_over_probability
            ),
            over_decimal_odds=(
                request.over_decimal_odds
            ),
            under_decimal_odds=(
                request.under_decimal_odds
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "model_version": MODEL_VERSION,
        "season": CURRENT_SEASON,
        **comparison,
    }
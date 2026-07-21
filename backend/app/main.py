import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.data_loader import load_match_data
from backend.app.model.calibration_service import (
    calibrate_prediction_totals,
)
from backend.app.model.config import (
    HALF_LIFE_DAYS,
    MODEL_VERSION,
    PRIOR_MATCHES,
    get_model_configuration,
)
from backend.app.model.market import (
    compare_two_way_market,
)
from backend.app.model.poisson import predict_match
from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


app = FastAPI(
    title="English Match Prediction API",
    description=(
        "API for an educational English football "
        "probability model"
    ),
    version=MODEL_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BACKTEST_SUMMARY_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "artifacts"
    / "backtest_summary.json"
)


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


# Load the historical data once when the API starts.
matches = load_match_data()


# Build the approved production model using settings from
# config.py. Experimental settings cannot accidentally enter
# production without changing the central configuration.
strength_model = build_team_strengths(
    matches=matches,
    prior_matches=PRIOR_MATCHES,
    half_life_days=HALF_LIFE_DAYS,
)


@app.get(
    "/",
    summary="API Home",
)
def home():
    return {
        "message": (
            "English Match Prediction API is running"
        ),
        "model_version": MODEL_VERSION,
        "documentation": "/docs",
    }


@app.get(
    "/health",
    summary="Health Check",
)
def health_check():
    return {
        "status": "healthy",
        "matches_loaded": len(matches),
        "teams_loaded": len(
            strength_model["teams"]
        ),
        "model": {
            "version": MODEL_VERSION,
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
    return get_model_configuration()


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
    summary="Available Teams",
)
def get_teams():
    teams = sorted(
        strength_model["teams"].index.tolist()
    )

    return {
        "teams": teams,
        "count": len(teams),
    }


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
        **calibrated_prediction,
    }


@app.post(
    "/predict/teams",
    summary="Team Match Prediction",
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

    return {
        "model_version": MODEL_VERSION,
        "home_team": request.home_team,
        "away_team": request.away_team,
        **calibrated_prediction,
    }


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
        **comparison,
    }
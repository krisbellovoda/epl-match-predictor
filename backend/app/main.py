from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.app.model.calibration_service import (
    calibrate_prediction_totals,
)

from backend.app.data_loader import load_match_data
from backend.app.model.market import compare_two_way_market
from backend.app.model.poisson import predict_match
from backend.app.model.team_strength import (
    build_team_strengths,
    estimate_expected_goals,
)


app = FastAPI(
    title="EPL Prediction API",
    description="API for an EPL Poisson prediction model",
    version="0.4.0",
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
        examples=["Arsenal"]
    )
    away_team: str = Field(
        examples=["Chelsea"]
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


matches = load_match_data()

strength_model = build_team_strengths(
    matches,
    prior_matches=5.0,
)


@app.get("/")
def home():
    return {
        "message": "EPL Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "matches_loaded": len(matches),
        "teams_loaded": len(
            strength_model["teams"]
        ),
        "model": {
            "prior_matches": (
                strength_model["prior_matches"]
            ),
            "half_life_days": (
                strength_model["half_life_days"]
            ),
        },
    }


@app.get("/teams")
def get_teams():
    teams = sorted(
        strength_model["teams"].index.tolist()
    )

    return {
        "teams": teams,
        "count": len(teams),
    }


@app.post("/predict")
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

    return calibrate_prediction_totals(
        prediction
    )


@app.post("/predict/teams")
def create_team_prediction(
    request: TeamPredictionRequest,
):
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
    prediction = calibrate_prediction_totals(
    prediction
)

    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        **prediction,
    }


@app.post("/market/compare")
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

    return comparison
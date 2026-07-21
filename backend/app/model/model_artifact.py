import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.app.model.config import (
    HALF_LIFE_DAYS,
    MODEL_VERSION,
    PRIOR_MATCHES,
)
from backend.app.model.team_strength import (
    build_team_strengths,
)


ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "team_strength_model.json"
)


def save_production_model(
    matches: pd.DataFrame,
    artifact_path: Path = ARTIFACT_PATH,
) -> dict:
    """
    Train and save the approved production team-strength model.
    """

    strength_model = build_team_strengths(
        matches=matches,
        prior_matches=PRIOR_MATCHES,
        half_life_days=HALF_LIFE_DAYS,
    )

    teams_frame = strength_model["teams"]

    serialized_teams = {
        str(team): {
            str(column): float(value)
            for column, value in row.items()
        }
        for team, row in teams_frame.iterrows()
    }

    last_match_date = None

    if "Date" in matches.columns and not matches.empty:
        last_match_date = str(
            matches["Date"].max()
        )

    artifact = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_matches": int(len(matches)),
        "last_training_match": last_match_date,
        "model": {
            "league_home_goals": strength_model[
                "league_home_goals"
            ],
            "league_away_goals": strength_model[
                "league_away_goals"
            ],
            "prior_matches": strength_model[
                "prior_matches"
            ],
            "half_life_days": strength_model[
                "half_life_days"
            ],
            "teams": serialized_teams,
        },
    }

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with artifact_path.open(
        "w",
        encoding="utf-8",
    ) as artifact_file:
        json.dump(
            artifact,
            artifact_file,
            indent=2,
        )

    return artifact


def load_production_model(
    artifact_path: Path = ARTIFACT_PATH,
) -> tuple[dict, dict]:
    """
    Load the approved production model without raw match data.

    Returns:
        A tuple containing the strength model and artifact
        metadata.
    """

    if not artifact_path.exists():
        raise FileNotFoundError(
            "Production model artifact was not found at "
            f"{artifact_path}"
        )

    with artifact_path.open(
        "r",
        encoding="utf-8",
    ) as artifact_file:
        artifact = json.load(artifact_file)

    if artifact.get("schema_version") != 1:
        raise ValueError(
            "Unsupported model artifact schema."
        )

    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Model artifact version does not match "
            f"production version {MODEL_VERSION}."
        )

    saved_model = artifact["model"]

    teams_frame = pd.DataFrame.from_dict(
        saved_model["teams"],
        orient="index",
    )

    strength_model = {
        "league_home_goals": float(
            saved_model["league_home_goals"]
        ),
        "league_away_goals": float(
            saved_model["league_away_goals"]
        ),
        "prior_matches": float(
            saved_model["prior_matches"]
        ),
        "half_life_days": (
            None
            if saved_model["half_life_days"] is None
            else float(
                saved_model["half_life_days"]
            )
        ),
        "teams": teams_frame,
    }

    metadata = {
        "schema_version": artifact[
            "schema_version"
        ],
        "model_version": artifact[
            "model_version"
        ],
        "generated_at": artifact[
            "generated_at"
        ],
        "training_matches": artifact[
            "training_matches"
        ],
        "last_training_match": artifact[
            "last_training_match"
        ],
    }

    return strength_model, metadata
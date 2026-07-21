import json
from pathlib import Path

from backend.app.model.probability_calibrator import (
    apply_probability_calibration,
)


ARTIFACT_FILE = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "over_2_5_calibrator.json"
)


def load_over_calibrator() -> dict:
    """
    Load the saved Over 2.5 calibration parameters.
    """

    if not ARTIFACT_FILE.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: "
            f"{ARTIFACT_FILE}"
        )

    with ARTIFACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as artifact:
        return json.load(artifact)


OVER_CALIBRATOR = load_over_calibrator()


def calibrate_prediction_totals(
    prediction: dict,
) -> dict:
    """
    Replace raw total-goals probabilities with calibrated
    probabilities while preserving the raw values.
    """

    raw_over_probability = prediction[
        "total_goals"
    ]["over_2_5"]

    raw_under_probability = prediction[
        "total_goals"
    ]["under_2_5"]

    calibrated_over_probability = float(
        apply_probability_calibration(
            [raw_over_probability],
            OVER_CALIBRATOR,
        )[0]
    )

    calibrated_under_probability = (
        1 - calibrated_over_probability
    )

    prediction["total_goals"] = {
        "over_2_5": calibrated_over_probability,
        "under_2_5": calibrated_under_probability,
        "raw_over_2_5": raw_over_probability,
        "raw_under_2_5": raw_under_probability,
        "calibrated": True,
    }

    return prediction
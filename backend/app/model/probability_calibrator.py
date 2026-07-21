import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    return np.log(
        clipped / (1 - clipped)
    )


def fit_probability_calibrator(
    predictions: pd.DataFrame,
) -> dict:
    """
    Fit logistic calibration using validation predictions.
    """

    probabilities = predictions[
        "over_2_5_probability"
    ].to_numpy(dtype=float)

    outcomes = predictions[
        "actual_over_2_5"
    ].to_numpy(dtype=float)

    logits = _logit(probabilities)

    def negative_log_loss(parameters):
        intercept, slope = parameters

        calibrated = expit(
            intercept + slope * logits
        )

        calibrated = np.clip(
            calibrated,
            1e-15,
            1 - 1e-15,
        )

        return -np.mean(
            outcomes * np.log(calibrated)
            + (1 - outcomes)
            * np.log(1 - calibrated)
        )

    result = minimize(
        negative_log_loss,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )

    if not result.success:
        raise RuntimeError(
            "Probability calibration failed."
        )

    intercept, slope = result.x

    return {
        "intercept": float(intercept),
        "slope": float(slope),
    }


def apply_probability_calibration(
    probabilities,
    calibrator: dict,
) -> np.ndarray:
    """
    Apply a fitted calibrator to new probabilities.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    logits = _logit(probability_array)

    calibrated = expit(
        calibrator["intercept"]
        + calibrator["slope"] * logits
    )

    return calibrated


def evaluate_binary_probabilities(
    probabilities,
    outcomes,
) -> dict:
    """
    Calculate Brier score and binary log loss.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    outcome_array = np.asarray(
        outcomes,
        dtype=float,
    )

    clipped = np.clip(
        probability_array,
        1e-15,
        1 - 1e-15,
    )

    brier_score = np.mean(
        (probability_array - outcome_array) ** 2
    )

    log_loss = -np.mean(
        outcome_array * np.log(clipped)
        + (1 - outcome_array)
        * np.log(1 - clipped)
    )

    return {
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
    }
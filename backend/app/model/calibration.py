import numpy as np
import pandas as pd


def calculate_over_calibration(
    predictions: pd.DataFrame,
    number_of_bins: int = 5,
) -> dict:
    """
    Compare predicted Over 2.5 probabilities with
    observed Over 2.5 frequencies.
    """

    required_columns = [
        "over_2_5_probability",
        "actual_over_2_5",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Predictions are missing: {missing_columns}"
        )

    calibration_data = predictions[
        required_columns
    ].copy()

    bin_edges = np.linspace(
        0,
        1,
        number_of_bins + 1,
    )

    calibration_data["probability_bin"] = pd.cut(
        calibration_data["over_2_5_probability"],
        bins=bin_edges,
        include_lowest=True,
    )

    grouped = calibration_data.groupby(
        "probability_bin",
        observed=False,
    ).agg(
        matches=("actual_over_2_5", "count"),
        average_prediction=(
            "over_2_5_probability",
            "mean",
        ),
        actual_frequency=(
            "actual_over_2_5",
            "mean",
        ),
    )

    grouped = grouped.dropna().reset_index()

    grouped["calibration_gap"] = (
        grouped["average_prediction"]
        - grouped["actual_frequency"]
    )

    total_matches = grouped["matches"].sum()

    expected_calibration_error = (
        (
            grouped["matches"]
            / total_matches
        )
        * grouped["calibration_gap"].abs()
    ).sum()

    rows = []

    for _, row in grouped.iterrows():
        rows.append(
            {
                "probability_range": str(
                    row["probability_bin"]
                ),
                "matches": int(row["matches"]),
                "average_prediction": float(
                    row["average_prediction"]
                ),
                "actual_frequency": float(
                    row["actual_frequency"]
                ),
                "calibration_gap": float(
                    row["calibration_gap"]
                ),
            }
        )

    return {
        "expected_calibration_error": float(
            expected_calibration_error
        ),
        "bins": rows,
    }
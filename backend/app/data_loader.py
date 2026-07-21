from pathlib import Path

import pandas as pd


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = BACKEND_DIRECTORY / "data" / "raw"

MATCH_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]

MARKET_COLUMNS = [
    "Avg>2.5",
    "Avg<2.5",
]

REQUIRED_COLUMNS = MATCH_COLUMNS + MARKET_COLUMNS


def load_match_data() -> pd.DataFrame:
    """
    Load, combine, clean, and sort all EPL CSV files.
    """

    data_files = sorted(
        DATA_DIRECTORY.glob("epl_*.csv")
    )

    if not data_files:
        raise FileNotFoundError(
            f"No EPL CSV files found in: {DATA_DIRECTORY}"
        )

    season_frames = []

    for data_file in data_files:
        season_matches = pd.read_csv(data_file)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in season_matches.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{data_file.name} is missing columns: "
                f"{missing_columns}"
            )

        season_matches = season_matches[
            REQUIRED_COLUMNS
        ].copy()

        season_matches["Season"] = (
            data_file.stem.replace("epl_", "")
        )

        season_frames.append(season_matches)

    matches = pd.concat(
        season_frames,
        ignore_index=True,
    )

    matches["Date"] = pd.to_datetime(
        matches["Date"],
        dayfirst=True,
        errors="coerce",
    )

    numeric_columns = [
        "FTHG",
        "FTAG",
        "Avg>2.5",
        "Avg<2.5",
    ]

    for column in numeric_columns:
        matches[column] = pd.to_numeric(
            matches[column],
            errors="coerce",
        )

    matches = matches.dropna(
        subset=MATCH_COLUMNS
    )

    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)

    matches = matches.sort_values("Date")
    matches = matches.reset_index(drop=True)

    return matches
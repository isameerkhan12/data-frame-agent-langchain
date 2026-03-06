"""
Data loader module for the LangChain DataFrame Agent.

Responsible for reading the weather CSV file and returning a clean
pandas DataFrame ready for the agent to analyse.

In the original custom implementation data loading was embedded directly
inside the main script.  Extracting it here follows the single-responsibility
principle and makes unit-testing straightforward.
"""

import logging
import os

import pandas as pd

from src.config import DATA_CSV_PATH

logger = logging.getLogger(__name__)


def load_weather_data(csv_path: str = DATA_CSV_PATH) -> pd.DataFrame:
    """Load weather data from a CSV file and return a cleaned DataFrame.

    The function resolves the path relative to the project root (one level
    above this file's directory) so that the project works regardless of
    the current working directory.

    Args:
        csv_path: Path to the CSV file.  Defaults to the value set in
                  ``src.config.DATA_CSV_PATH`` which itself reads from the
                  ``DATA_CSV_PATH`` environment variable.

    Returns:
        A pandas DataFrame containing the weather records.

    Raises:
        FileNotFoundError: When the CSV file does not exist at ``csv_path``.
        ValueError: When the CSV file is empty or cannot be parsed.
    """
    # Resolve relative paths against the project root so the script can be
    # run from any working directory.
    if not os.path.isabs(csv_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Weather data CSV not found at '{csv_path}'. "
            "Please ensure the file exists or set the DATA_CSV_PATH "
            "environment variable to the correct path."
        )

    logger.info("Loading weather data from '%s' …", csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file '{csv_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV file '{csv_path}' is empty.")

    # Parse the 'date' column to datetime for time-based queries.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        nat_count = df["date"].isna().sum()
        if nat_count > 0:
            logger.warning(
                "%d date value(s) could not be parsed and were set to NaT.", nat_count
            )

    logger.info(
        "Loaded %d rows and %d columns from '%s'.",
        len(df),
        len(df.columns),
        csv_path,
    )
    logger.debug("Columns: %s", list(df.columns))

    return df

"""
load_data.py

Utilities for loading NYC Yellow Taxi trip data and
Taxi Zone Lookup data.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"



# ---------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------

def load_month(month: str) -> pd.DataFrame:
    """
    Load a single month's Yellow Taxi data.

    Parameters
    ----------
    month : str
        Month in YYYY-MM format (e.g. "2023-01").

    Returns
    -------
    pd.DataFrame
        Yellow Taxi dataframe for the requested month.
    """
    file_path = RAW_DATA_PATH / f"yellow_tripdata_{month}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_parquet(file_path)

def load_featured_data():

    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "featured_data.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_parquet(file_path)

def load_all_months() -> pd.DataFrame:
    """
    Load January–June 2023 and concatenate them.

    Returns
    -------
    pd.DataFrame
    """

    months = [
        "2023-01",
        "2023-02",
        "2023-03",
        "2023-04",
        "2023-05",
        "2023-06",
    ]

    dfs = [load_month(month) for month in months]

    return pd.concat(dfs, ignore_index=True)


def load_zone_lookup() -> pd.DataFrame:
    """
    Load Taxi Zone Lookup table.

    Returns
    -------
    pd.DataFrame
    """

    lookup_path = RAW_DATA_PATH / "taxi_zone_lookup.csv"

    if not lookup_path.exists():
        raise FileNotFoundError(f"File not found: {lookup_path}")

    return pd.read_csv(lookup_path)


# ---------------------------------------------------------------------
# Script Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    taxi_df = load_all_months()
    zone_df = load_zone_lookup()

    print("=" * 60)
    print("NYC Taxi Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Taxi Records : {len(taxi_df):,}")
    print(f"Taxi Columns : {taxi_df.shape[1]}")
    print(f"Taxi Zones   : {len(zone_df)}")


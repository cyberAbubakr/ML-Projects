"""
build_dataset.py

Purpose:
--------
This module loads the raw NYC Taxi dataset, cleans it, creates the target
variable for machine learning, removes invalid records and data leakage,
and saves a cleaned dataset for the next stage of the ML pipeline.

Author: Abubakr Kazmi
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_data import load_all_months


def build_dataset():
    """
    Complete preprocessing pipeline.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """

    # ==========================================================
    # Step 1: Load merged dataset
    # ==========================================================
    print("Loading dataset...")
    df = load_all_months()

    print(f"Original Shape: {df.shape}")

    # ==========================================================
    # Step 2: Create target variable
    # Trip duration in minutes
    # ==========================================================
    print("Creating target variable...")

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] -
        df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # ==========================================================
    # Step 3: Create log-transformed target
    # Used for comparison with the original target
    # ==========================================================
    df["log_trip_duration"] = np.log1p(df["trip_duration_minutes"])

    # ==========================================================
    # Step 4: Remove invalid timestamps
    # Remove corrupted records (e.g., year 2001)
    # ==========================================================
    print("Removing invalid timestamps...")

    before = len(df)

    df = df[df["tpep_pickup_datetime"].dt.year >= 2023]
    df = df[df["tpep_dropoff_datetime"].dt.year >= 2023]
    df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]]

    print(f"Removed {before - len(df):,} rows")

    # ==========================================================
    # Step 5: Remove invalid trip durations
    # Keep trips between 1 minute and 6 hours
    # ==========================================================
    print("Cleaning trip durations...")

    before = len(df)

    df = df[df["trip_duration_minutes"] > 0]
    df = df[df["trip_duration_minutes"] <= 360]

    print(f"Removed {before - len(df):,} rows")

    # ==========================================================
    # Step 6: Remove invalid passenger counts
    # NYC taxis normally carry between 1 and 6 passengers
    # ==========================================================
    print("Cleaning passenger counts...")

    before = len(df)

    df = df[df["passenger_count"].between(1, 6)]

    print(f"Removed {before - len(df):,} rows")

    # ==========================================================
    # Step 7: Remove invalid trip distances
    # Distance must be greater than zero
    # ==========================================================
    print("Cleaning trip distances...")

    before = len(df)

    df = df[df["trip_distance"] > 0]

    print(f"Removed {before - len(df):,} rows")

    # ==========================================================
    # Step 8: Remove data leakage
    # These values are only known AFTER the trip finishes.
    # They should never be used for ETA prediction.
    # ==========================================================
    print("Removing leakage columns...")

    leakage_columns = [
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
    ]

    df = df.drop(columns=leakage_columns, errors="ignore")

    # ==========================================================
    # Step 9: Remove duplicate rows (extra safety check)
    # ==========================================================
    before = len(df)

    df = df.drop_duplicates()

    print(f"Duplicate rows removed: {before - len(df):,}")

    # ==========================================================
    # Step 10: Save cleaned dataset
    # ==========================================================
    output_folder = Path("data/interim")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "cleaned_data.parquet"

    df.to_parquet(output_file, index=False)

    print(f"\nCleaned dataset saved to: {output_file}")
    print(f"Final Shape: {df.shape}")

    return df


# ==============================================================
# Main Program
# ==============================================================

if __name__ == "__main__":

    cleaned_df = build_dataset()

    print("\nDataset Preview")
    print(cleaned_df.head())
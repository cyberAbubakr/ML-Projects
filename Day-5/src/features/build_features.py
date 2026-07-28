"""
build_features.py

Purpose
-------
Create machine learning features from the cleaned NYC Taxi dataset.

Author: Abubakr Kazmi
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def load_featured_data() -> pd.DataFrame:
    """
    Load the feature-engineered dataset.
    """

    file_path = PROJECT_ROOT / "data" / "processed" / "featured_data.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_parquet(file_path)

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_cleaned_data():
    file_path = INTERIM_DATA_PATH / "cleaned_data.parquet"

    logger.info("Loading cleaned dataset...")
    df = pd.read_parquet(file_path)

    return df


def load_zone_lookup() -> pd.DataFrame:
    """
    Load Taxi Zone Lookup table.
    """

    lookup_path = RAW_DATA_PATH / "taxi_zone_lookup.csv"

    logger.info("Loading Taxi Zone Lookup...")

    zone_lookup = pd.read_csv(lookup_path)

    zone_lookup.columns = (
        zone_lookup.columns
        .str.strip()
        .str.lower()
    )

    return zone_lookup


# =============================================================================
# Time Features
# =============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    """

    logger.info("Creating time features...")

    pickup = df["tpep_pickup_datetime"]

    df["pickup_year"] = pickup.dt.year.astype("int16")
    df["pickup_month"] = pickup.dt.month.astype("int8")
    df["pickup_day"] = pickup.dt.day.astype("int8")

    df["pickup_hour"] = pickup.dt.hour.astype("int8")

    df["pickup_dayofweek"] = (
        pickup.dt.dayofweek.astype("int8")
    )

    df["pickup_week"] = (
        pickup.dt.isocalendar().week.astype("int16")
    )

    df["pickup_weekend"] = (
        df["pickup_dayofweek"] >= 5
    ).astype("int8")

    df["pickup_quarter"] = (
        pickup.dt.quarter.astype("int8")
    )

    return df


# =============================================================================
# Rush Hour Features
# =============================================================================

def create_rush_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rush hour indicator.
    """

    logger.info("Creating rush hour features...")

    morning_peak = df["pickup_hour"].between(7, 9)

    evening_peak = df["pickup_hour"].between(16, 19)

    df["rush_hour"] = (
        morning_peak | evening_peak
    ).astype("int8")

    return df


# =============================================================================
# Cyclical Time Features
# =============================================================================

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical time variables.
    """

    logger.info("Creating cyclical features...")

    df["hour_sin"] = np.sin(
        2 * np.pi * df["pickup_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["pickup_hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["pickup_dayofweek"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["pickup_dayofweek"] / 7
    )

    return df


# =============================================================================
# Location Features
# =============================================================================

def create_location_features(
    df: pd.DataFrame,
    zone_lookup: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge borough information into dataset.
    """

    logger.info("Creating location features...")

    pickup_lookup = zone_lookup[
        ["locationid", "borough"]
    ].rename(
        columns={
            "locationid": "PULocationID",
            "borough": "pickup_borough"
        }
    )

    dropoff_lookup = zone_lookup[
        ["locationid", "borough"]
    ].rename(
        columns={
            "locationid": "DOLocationID",
            "borough": "dropoff_borough"
        }
    )

    df = df.merge(
        pickup_lookup,
        on="PULocationID",
        how="left"
    )

    df = df.merge(
        dropoff_lookup,
        on="DOLocationID",
        how="left"
    )

    return df

# =============================================================================
# Route Features
# =============================================================================

def create_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create route-based features.
    """

    logger.info("Creating route features...")

    df["same_borough"] = (
        df["pickup_borough"] ==
        df["dropoff_borough"]
    ).astype("int8")

    df["same_location"] = (
        df["PULocationID"] ==
        df["DOLocationID"]
    ).astype("int8")

    return df


# =============================================================================
# Airport Features
# =============================================================================

def create_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create airport trip indicator.

    NYC Taxi Zone IDs:
        JFK = 132
        Newark = 1
        LaGuardia = 138
    """

    logger.info("Creating airport features...")

    airport_ids = [1, 132, 138]

    pickup_airport = df["PULocationID"].isin(airport_ids)

    dropoff_airport = df["DOLocationID"].isin(airport_ids)

    df["is_airport_trip"] = (
        pickup_airport | dropoff_airport
    ).astype("int8")

    return df


# =============================================================================
# Distance Features
# =============================================================================

def create_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered distance features.
    """

    logger.info("Creating distance features...")

    df["distance_squared"] = (
        df["trip_distance"] ** 2
    ).astype("float32")

    df["log_trip_distance"] = np.log1p(
        df["trip_distance"]
    ).astype("float32")

    df["distance_per_passenger"] = (
        df["trip_distance"] /
        df["passenger_count"].replace(0, 1)
    ).astype("float32")

    return df


# =============================================================================
# Memory Optimization
# =============================================================================

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by converting columns to
    smaller numeric data types where appropriate.
    """

    logger.info("Optimizing dataframe memory usage...")

    int8_cols = [
        "passenger_count",
        "payment_type",
        "trip_type",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_weekend",
        "pickup_quarter",
        "rush_hour",
        "same_borough",
        "same_location",
        "is_airport_trip",
    ]

    int16_cols = [
        "pickup_year",
        "pickup_week",
    ]

    float32_cols = [
        "trip_distance",
        "trip_duration_minutes",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "distance_squared",
        "log_trip_distance",
        "distance_per_passenger",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    for col in int16_cols:
        if col in df.columns:
            df[col] = df[col].astype("int16")

    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    before = df.memory_usage(deep=True).sum() / 1024**2

    logger.info(f"Optimized dataframe size: {before:.2f} MB")

    return df


# =============================================================================
# Drop Unused Columns
# =============================================================================

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are no longer needed after
    feature engineering.
    """

    logger.info("Dropping unused columns...")

    columns_to_drop = [
        "pickup_borough",
        "dropoff_borough",
    ]

    existing = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df = df.drop(columns=existing)

    return df


# =============================================================================
# Save Featured Dataset
# =============================================================================

def save_featured_dataset(df: pd.DataFrame) -> None:
    """
    Save the engineered dataset.
    """

    output_path = (
        PROCESSED_DATA_PATH /
        "featured_taxi_data.parquet"
    )

    logger.info("Saving featured dataset...")

    df.to_parquet(
        output_path,
        index=False,
        compression="snappy"
    )

    logger.info(f"Dataset saved to:\n{output_path}")

    # =============================================================================
# Build Feature Pipeline
# =============================================================================

def build_features() -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset.
    """

    logger.info("=" * 70)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    df = load_cleaned_data()
    zone_lookup = load_zone_lookup()

    # -------------------------------------------------------------------------
    # Time Features
    # -------------------------------------------------------------------------
    df = create_time_features(df)

    # -------------------------------------------------------------------------
    # Rush Hour Features
    # -------------------------------------------------------------------------
    df = create_rush_hour_features(df)

    # -------------------------------------------------------------------------
    # Cyclical Features
    # -------------------------------------------------------------------------
    df = create_cyclical_features(df)

    # -------------------------------------------------------------------------
    # Location Features
    # -------------------------------------------------------------------------
    df = create_location_features(df, zone_lookup)

    # -------------------------------------------------------------------------
    # Route Features
    # -------------------------------------------------------------------------
    df = create_route_features(df)

    # -------------------------------------------------------------------------
    # Airport Features
    # -------------------------------------------------------------------------
    df = create_airport_features(df)

    # -------------------------------------------------------------------------
    # Distance Features
    # -------------------------------------------------------------------------
    df = create_distance_features(df)

    # -------------------------------------------------------------------------
    # Remove temporary columns
    # -------------------------------------------------------------------------
    df = drop_unused_columns(df)

    # -------------------------------------------------------------------------
    # Optimize memory usage
    # -------------------------------------------------------------------------
    df = optimize_dataframe(df)

    # -------------------------------------------------------------------------
    # Save engineered dataset
    # -------------------------------------------------------------------------
    save_featured_dataset(df)

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    logger.info(f"Final Shape : {df.shape}")
    logger.info(
        f"Memory Usage : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    featured_df = build_features()

    print("\n" + "=" * 70)
    print("Feature Engineering Pipeline Completed Successfully!")
    print("=" * 70)
    print(f"Final Dataset Shape : {featured_df.shape}")

    print("\nFeature Columns:\n")
    for column in featured_df.columns:
        print(f"• {column}")
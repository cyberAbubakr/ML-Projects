"""
preprocessing.py

Reusable data preprocessing pipeline for the Customer Personality Analysis
dataset (Module 3 - Data Preprocessing).

Each pipeline step is implemented as an independent, well-documented
function so it can be reused from a notebook, another script, or run
end-to-end via main(). Functions never mutate the DataFrame they receive;
each one returns a new copy so steps can be tested or reordered safely.

Usage (as a script):
    python preprocessing.py

Usage (as a module, e.g. from a notebook):
    from preprocessing import load_data, handle_missing_values, ...
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------
PROJECT_DIR = Path.cwd()
RAW_DATA_DIR = PROJECT_DIR / "01_Raw_Data"
CLEANED_DATA_DIR = PROJECT_DIR / "03_Cleaned_Data"
REPORT_DIR = PROJECT_DIR / "04_Preprocessing_Report"

DATASET_PATH = RAW_DATA_DIR / "marketing_campaign.csv"
CLEANED_DATASET_PATH = CLEANED_DATA_DIR / "customer_personality_cleaned.csv"

# Latest customer enrollment year found in the dataset (Module 1 finding).
# Used as the reference point for age and date-related calculations.
CURRENT_YEAR = 2014

# Columns that should never contain a negative or wildly extreme value.
SPENDING_COLUMNS = [
    "Income",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

CONSTANT_COLUMNS = ["Z_CostContact", "Z_Revenue"]


# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
def load_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the raw dataset from a tab-separated CSV file."""
    df = pd.read_csv(path, sep="\t")
    return df


# ------------------------------------------------------------
# 2. Handle Missing Values
# ------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Income values using the column median.

    The median is used instead of the mean because Income is
    right-skewed (a few very high earners), and the median is far
    less sensitive to those extreme values.
    """
    df = df.copy()
    median_income = df["Income"].median()
    df["Income"] = df["Income"].fillna(median_income)
    return df


# ------------------------------------------------------------
# 3. Remove Duplicate Records
# ------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows and duplicated customer IDs."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="ID", keep="first")
    return df


# ------------------------------------------------------------
# 4. Correct Data Types
# ------------------------------------------------------------
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate, memory-efficient data types."""
    df = df.copy()

    categorical_cols = ["Education", "Marital_Status"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    binary_cols = [
        "Kidhome", "Teenhome",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5",
        "Complain", "Response",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    return df


# ------------------------------------------------------------
# 5. Convert Dates
# ------------------------------------------------------------
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Dt_Customer to datetime and extract Enrollment_Year/Month.

    Rows where the date cannot be parsed become NaT and are dropped,
    since an unparseable enrollment date cannot be trusted downstream.
    """
    df = df.copy()
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    df = df.dropna(subset=["Dt_Customer"])

    df["Enrollment_Year"] = df["Dt_Customer"].dt.year
    df["Enrollment_Month"] = df["Dt_Customer"].dt.month
    return df


# ------------------------------------------------------------
# 6. Standardize Categories
# ------------------------------------------------------------
def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize text categories: strip spaces, title case, merge typos."""
    df = df.copy()

    df["Education"] = df["Education"].astype(str).str.strip().str.title()
    df["Marital_Status"] = (
        df["Marital_Status"].astype(str).str.strip().str.title()
    )

    # Known inconsistent / invalid categories are folded into "Single".
    marital_status_map = {
        "Alone": "Single",
        "Yolo": "Single",
        "Absurd": "Single",
    }
    df["Marital_Status"] = df["Marital_Status"].replace(marital_status_map)

    df["Education"] = df["Education"].astype("category")
    df["Marital_Status"] = df["Marital_Status"].astype("category")
    return df


# ------------------------------------------------------------
# 7. Validate Customer Age
# ------------------------------------------------------------
def validate_age(
    df: pd.DataFrame, min_age: int = 18, max_age: int = 100
) -> pd.DataFrame:
    """Create an Age column and remove records with unrealistic ages."""
    df = df.copy()
    df["Age"] = CURRENT_YEAR - df["Year_Birth"]

    invalid_mask = (
        (df["Age"] > max_age)
        | (df["Age"] < min_age)
        | (df["Year_Birth"] > CURRENT_YEAR)
    )
    df = df.loc[~invalid_mask].copy()
    return df


# ------------------------------------------------------------
# 8. Detect Outliers
# ------------------------------------------------------------
def detect_outliers(
    df: pd.DataFrame, columns: list = None
) -> pd.DataFrame:
    """Return an outlier summary DataFrame using the IQR method."""
    if columns is None:
        columns = SPENDING_COLUMNS

    records = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        records.append({
            "Variable": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Outlier Count": int(outlier_count),
        })

    return pd.DataFrame(records)


# ------------------------------------------------------------
# 9. Handle Outliers
# ------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, column: str = "Income") -> pd.DataFrame:
    """Winsorize a column by clipping values to its IQR bounds.

    Only Income is winsorized. Spending columns (MntWines, MntMeatProducts,
    etc.) are left untouched because large purchase amounts are plausible
    real behavior, not data errors, and clipping them would remove
    genuine high-value customer signal.
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


# ------------------------------------------------------------
# 10. Remove Constant Features
# ------------------------------------------------------------
def remove_constant_columns(
    df: pd.DataFrame, columns: list = None
) -> pd.DataFrame:
    """Drop columns whose value never changes across the dataset.

    Z_CostContact and Z_Revenue are constant for every customer, so they
    carry no analytical or predictive value and are safe to remove.
    """
    if columns is None:
        columns = CONSTANT_COLUMNS

    df = df.copy()
    cols_to_drop = [c for c in columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


# ------------------------------------------------------------
# 11. Final Validation
# ------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run a final set of sanity checks and return a checklist DataFrame."""
    checks = {
        "No missing values in Income": df["Income"].isnull().sum() == 0,
        "No duplicate rows": df.duplicated().sum() == 0,
        "No duplicate IDs": df["ID"].duplicated().sum() == 0,
        "Dt_Customer is datetime": pd.api.types.is_datetime64_any_dtype(
            df["Dt_Customer"]
        ),
        "All ages between 18 and 100": df["Age"].between(18, 100).all(),
        "No constant Z_ columns": not any(
            c in df.columns for c in CONSTANT_COLUMNS
        ),
    }
    result = pd.DataFrame({
        "Check": list(checks.keys()),
        "Passed": list(checks.values()),
    })
    return result


# ------------------------------------------------------------
# 13. Save Clean Dataset
# ------------------------------------------------------------
def save_dataset(
    df: pd.DataFrame, path: Path = CLEANED_DATASET_PATH
) -> Path:
    """Save the cleaned dataset to CSV, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------
# Full Pipeline
# ------------------------------------------------------------
def main() -> pd.DataFrame:
    """Run the complete preprocessing pipeline end-to-end."""
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_data_types(df)
    df = convert_dates(df)
    df = clean_categories(df)
    df = validate_age(df)

    outlier_summary = detect_outliers(df, SPENDING_COLUMNS)
    print("\nOutlier summary (IQR method):")
    print(outlier_summary.to_string(index=False))

    df = handle_outliers(df, column="Income")
    df = remove_constant_columns(df)

    checklist = validate_dataset(df)
    print("\nFinal validation checklist:")
    print(checklist.to_string(index=False))

    output_path = save_dataset(df)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    main()

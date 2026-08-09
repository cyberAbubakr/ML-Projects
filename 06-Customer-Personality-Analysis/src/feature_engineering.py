"""feature_engineering.py — Module 5: Feature Engineering & Data Preprocessing.

Reusable pipeline that turns the Module 3 cleaned dataset into a machine
learning-ready dataset for customer segmentation (clustering). Reuses
preprocessing.handle_missing_values() and eda_utils.load_cleaned_data() /
add_engineered_columns() instead of duplicating that logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from preprocessing import handle_missing_values
from eda_utils import load_cleaned_data, add_engineered_columns, CAMPAIGN_COLS

PROJECT_DIR = Path.cwd()
FEATURES_PATH = PROJECT_DIR / "03_Cleaned_Data" / "customer_personality_features.csv"

REFERENCE_DATE = pd.Timestamp("2014-12-31")

IDENTIFIER_COLS = ["ID"]
REDUNDANT_COLS = ["Year_Birth", "Dt_Customer", "Enrollment_Year", "Enrollment_Month"]

CATEGORICAL_NOMINAL_COLS = [
    "Education", "Marital_Status", "Preferred_Shopping_Channel", "Product_Preference",
]
ACTIVITY_LEVEL_MAP = {"Inactive": 0, "Moderate": 1, "Active": 2}


# ------------------------------------------------------------
# Task 1: Customer Feature Creation
# ------------------------------------------------------------
def add_customer_tenure(df):
    """Add Customer_Tenure (days since enrollment, relative to 2014-12-31)."""
    df = df.copy()
    enrollment_date = pd.to_datetime(df["Dt_Customer"])
    df["Customer_Tenure"] = (REFERENCE_DATE - enrollment_date).dt.days
    return df


def add_family_features(df):
    """Add Total_Children and Family_Size (children + self + partner)."""
    df = df.copy()
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    is_partnered = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
    df["Family_Size"] = df["Total_Children"] + 1 + is_partnered
    return df


def add_campaign_acceptance(df):
    """Add Total_Campaign_Acceptance (count of accepted campaigns, incl. Response)."""
    df = df.copy()
    df["Total_Campaign_Acceptance"] = df[CAMPAIGN_COLS + ["Response"]].sum(axis=1)
    return df


def add_spending_ratios(df):
    """Add Avg_Spending_Per_Purchase and Deal_Dependency (safe division by zero)."""
    df = df.copy()
    df["Avg_Spending_Per_Purchase"] = (
        df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    df["Deal_Dependency"] = (
        df["NumDealsPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    return df


def add_digital_engagement(df):
    """Add Digital_Engagement (web purchases + monthly web visits)."""
    df = df.copy()
    df["Digital_Engagement"] = df["NumWebPurchases"] + df["NumWebVisitsMonth"]
    return df


def add_preferred_channel(df):
    """Add Preferred_Shopping_Channel (channel with the most purchases)."""
    df = df.copy()
    channels = df[["NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"]]
    df["Preferred_Shopping_Channel"] = channels.idxmax(axis=1).str.replace(
        "Num|Purchases", "", regex=True
    )
    return df


def add_product_preference(df):
    """Add Product_Preference (product category with the highest spending)."""
    df = df.copy()
    mnt_map = {
        "MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
    }
    df["Product_Preference"] = df[list(mnt_map)].idxmax(axis=1).map(mnt_map)
    return df


def add_activity_level(df):
    """Add Customer_Activity_Level (Active/Moderate/Inactive from Recency)."""
    df = df.copy()
    df["Customer_Activity_Level"] = pd.cut(
        df["Recency"], bins=[-1, 30, 60, df["Recency"].max()],
        labels=["Active", "Moderate", "Inactive"], include_lowest=True,
    ).astype(str)
    return df


# ------------------------------------------------------------
# Task 2: Categorical Feature Encoding
# ------------------------------------------------------------
def encode_categoricals(df, columns=CATEGORICAL_NOMINAL_COLS):
    """One-hot encode nominal columns; label-encode ordinal Customer_Activity_Level."""
    df = df.copy()
    df["Customer_Activity_Level"] = df["Customer_Activity_Level"].map(ACTIVITY_LEVEL_MAP)
    df = pd.get_dummies(df, columns=columns, prefix=columns)
    dummy_cols = [c for c in df.columns if df[c].dtype == "bool"]
    df[dummy_cols] = df[dummy_cols].astype(int)
    return df


# ------------------------------------------------------------
# Task 3: Feature Selection
# ------------------------------------------------------------
def select_features(df):
    """Drop identifiers and columns made redundant by newly engineered features."""
    df = df.copy()
    cols_to_drop = [c for c in IDENTIFIER_COLS + REDUNDANT_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


# ------------------------------------------------------------
# Task 4: Skewness and Feature Transformation
# ------------------------------------------------------------
def transform_skewed(df, columns, threshold=0.75):
    """Log1p-transform numeric columns with |skew| above threshold. Returns df, summary."""
    df = df.copy()
    records = []
    for col in columns:
        before = skew(df[col].dropna())
        transformed = False
        if abs(before) > threshold and (df[col] >= 0).all():
            df[col] = np.log1p(df[col])
            transformed = True
        after = skew(df[col].dropna())
        records.append({
            "Feature": col, "Skew_Before": round(before, 3),
            "Skew_After": round(after, 3), "Transformed": transformed,
        })
    return df, pd.DataFrame(records)


# ------------------------------------------------------------
# Task 5: Feature Scaling
# ------------------------------------------------------------
def scale_features(df, columns, method="standard"):
    """Scale numeric columns in place using StandardScaler, MinMaxScaler, or RobustScaler."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
    scaler = scalers[method]
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# ------------------------------------------------------------
# Task 8: Final Dataset Validation
# ------------------------------------------------------------
def validate_dataset(df):
    """Run final checks confirming the dataset is ready for clustering."""
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    duplicate_count = int(df.duplicated().sum())
    checks = {
        "No missing values": df.isnull().sum().sum() == 0,
        "Duplicate rows (post ID removal)": duplicate_count,
        "All columns numeric": len(non_numeric) == 0,
        "No identifier columns present": "ID" not in df.columns,
    }
    return pd.DataFrame({"Check": list(checks.keys()), "Result": list(checks.values())})


# ------------------------------------------------------------
# Pipeline Orchestration
# ------------------------------------------------------------
def build_features():
    """Run the complete Task 1-5 feature engineering pipeline. Returns (df, skew_summary)."""
    df = load_cleaned_data()
    df = handle_missing_values(df)
    df = add_engineered_columns(df)  # Total_Spending, Total_Purchases
    df = add_customer_tenure(df)
    df = add_family_features(df)
    df = add_campaign_acceptance(df)
    df = add_spending_ratios(df)
    df = add_digital_engagement(df)
    df = add_preferred_channel(df)
    df = add_product_preference(df)
    df = add_activity_level(df)
    df = select_features(df)
    df = encode_categoricals(df)

    skewed_candidates = [
        "Income", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "Total_Spending", "Avg_Spending_Per_Purchase",
    ]
    df, skew_summary = transform_skewed(df, skewed_candidates)

    binary_like = [c for c in df.columns if df[c].nunique() <= 2]
    scale_cols = [c for c in df.select_dtypes(include="number").columns if c not in binary_like]
    df = scale_features(df, scale_cols, method="standard")

    return df, skew_summary


def save_features(df, path=FEATURES_PATH):
    """Save the final engineered dataset without touching the cleaned CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main():
    df, skew_summary = build_features()
    path = save_features(df)
    validation = validate_dataset(df)
    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")
    print(validation.to_string(index=False))
    return df, skew_summary, validation


if __name__ == "__main__":
    main()

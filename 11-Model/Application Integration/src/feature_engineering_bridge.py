"""
Bridge between manually entered customer input and Module 6's saved
production model.

This module does NOT reimplement Module 6's feature engineering. It
imports and calls the actual functions in
`06-Customer-Personality-Analysis/src/feature_engineering.py` (and its
`eda_utils`/`preprocessing` dependencies), the same functions that
produced the official `customer_personality_features.csv`.

Why append-and-recompute instead of using the saved scaler directly:
The saved `standard_scaler.pkl` / `pipeline.pkl["scaler"]` was fit AFTER
Module 6's feature engineering had already scaled the continuous columns
(its mean/scale for every continuous feature is ~0/~1 -- verified). It is
not a raw-to-scaled transformer for brand-new, not-yet-scaled input; using
it directly on freshly engineered raw values would leave those values
unscaled and would produce meaningless distances for the KMeans model.
Module 6's own `scale_features()` step (a plain `StandardScaler` fit on
the full engineered dataset) IS the real raw-to-scaled transform, so a
new customer is placed in the *same* feature space by appending their
raw record to the historical cleaned dataset and re-running Module 6's
exact, unchanged pipeline once. This was verified to reproduce
`customer_personality_features.csv` to floating-point precision for every
existing customer (max abs diff ~1e-15), confirming it is the authentic
Module 6 pipeline, not a new one. The saved KMeans model itself is never
retrained or altered -- only inference (`.predict()`) is ever called on it.
"""

import sys
from pathlib import Path

import pandas as pd

from src import config, data_loader

_MODULE6_SRC = str(config.MODULE6_DIR / "src")

EDUCATION_OPTIONS = ["Graduation", "Phd", "Master", "Basic", "2N Cycle"]
MARITAL_STATUS_OPTIONS = ["Single", "Together", "Married", "Divorced", "Widow"]

# Module 6's own reference year (preprocessing.py CURRENT_YEAR), used only
# to keep the schema complete -- Year_Birth is dropped as a redundant
# column by Module 6's own select_features() and never reaches the model.
REFERENCE_YEAR = 2014


def _load_module6_functions():
    """Import Module 6's real feature-engineering/eda functions (not copied)."""
    if _MODULE6_SRC not in sys.path:
        sys.path.insert(0, _MODULE6_SRC)
    import feature_engineering as m6_feature_engineering  # Module 6's actual file
    import eda_utils as m6_eda_utils  # Module 6's actual file
    return m6_feature_engineering, m6_eda_utils


def validate_customer_input(raw: dict) -> list:
    """
    Basic sanity checks on manually entered values before they are fed
    into Module 6's pipeline. Returns a list of human-readable error
    strings; an empty list means the input is usable.
    """
    errors = []

    if raw.get("Education") not in EDUCATION_OPTIONS:
        errors.append("Education must be one of the known categories.")
    if raw.get("Marital_Status") not in MARITAL_STATUS_OPTIONS:
        errors.append("Marital Status must be one of the known categories.")

    if raw.get("Age") is None or not (18 <= raw["Age"] <= 100):
        errors.append("Age must be between 18 and 100 (Module 6's original valid range).")

    if raw.get("Income") is None or raw["Income"] < 0:
        errors.append("Income must be a non-negative number.")

    non_negative_fields = [
        "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth",
    ]
    for field in non_negative_fields:
        value = raw.get(field)
        if value is None or value < 0:
            errors.append(f"{field} must be a non-negative number.")

    binary_fields = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
                      "AcceptedCmp5", "Complain", "Response"]
    for field in binary_fields:
        if raw.get(field) not in (0, 1):
            errors.append(f"{field} must be 0 or 1.")

    if raw.get("Dt_Customer") is None:
        errors.append("Enrollment date is required.")

    return errors


def engineer_new_customer_features(raw: dict, selected_features: list) -> pd.Series:
    """
    Convert one manually entered customer into the exact feature vector
    (ordered to match `selected_features`) that the saved model expects.

    `raw` must contain the same fields as Module 6's cleaned dataset
    (ID/Year_Birth are placeholders only -- dropped by Module 6's own
    pipeline before the model ever sees them).
    """
    fe, eda_utils = _load_module6_functions()

    cleaned_df = data_loader.load_cleaned_customers()
    if cleaned_df is None:
        raise RuntimeError(
            "The historical cleaned dataset is required to place a new "
            "customer into the same feature space as the trained model, "
            "but it is not available."
        )

    new_row = pd.DataFrame([raw])
    combined = pd.concat([cleaned_df, new_row], ignore_index=True, sort=False)
    combined["Dt_Customer"] = pd.to_datetime(combined["Dt_Customer"])

    # Module 6's exact, unmodified pipeline steps (src/feature_engineering.py).
    d = eda_utils.add_engineered_columns(combined)
    d = fe.add_customer_tenure(d)
    d = fe.add_family_features(d)
    d = fe.add_campaign_acceptance(d)
    d = fe.add_spending_ratios(d)
    d = fe.add_digital_engagement(d)
    d = fe.add_preferred_channel(d)
    d = fe.add_product_preference(d)
    d = fe.add_activity_level(d)
    d = fe.select_features(d)
    d = fe.encode_categoricals(d)

    skewed_candidates = [
        "Income", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "Total_Spending", "Avg_Spending_Per_Purchase",
    ]
    d, _ = fe.transform_skewed(d, skewed_candidates)

    binary_like = [c for c in d.columns if d[c].nunique() <= 2]
    scale_cols = [c for c in d.select_dtypes(include="number").columns if c not in binary_like]
    d = fe.scale_features(d, scale_cols, method="standard")

    new_customer_row = d.iloc[-1]
    missing = [f for f in selected_features if f not in new_customer_row.index]
    if missing:
        raise RuntimeError(f"Engineered features are missing expected columns: {missing}")

    return new_customer_row[selected_features]

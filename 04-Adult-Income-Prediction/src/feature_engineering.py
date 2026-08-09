import numpy as np
import pandas as pd


# ==========================================================
# Age Groups
# ==========================================================

def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age categories from the age feature.
    """

    bins = [0, 25, 40, 60, 100]
    labels = [
        "Young Adult",
        "Adult",
        "Middle Aged",
        "Senior"
    ]

    df = df.copy()

    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


# ==========================================================
# Capital Net
# ==========================================================

def create_capital_net(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create net capital feature.
    """

    df = df.copy()

    df["CapitalNet"] = (
        df["capital_gain"] -
        df["capital_loss"]
    )

    return df


# ==========================================================
# Work Hours Category
# ==========================================================

def create_work_hours_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize working hours.
    """

    bins = [0, 34, 40, 60, np.inf]

    labels = [
        "Part-Time",
        "Full-Time",
        "Overtime",
        "Heavy Overtime"
    ]

    df = df.copy()

    df["WorkHoursCategory"] = pd.cut(
        df["hours_per_week"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


# ==========================================================
# Education Level
# ==========================================================

def create_education_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group education categories into broader levels.
    """

    mapping = {

        "Preschool": "School",

        "1st-4th": "School",
        "5th-6th": "School",
        "7th-8th": "School",
        "9th": "School",
        "10th": "School",
        "11th": "School",
        "12th": "School",

        "HS-grad": "High School",

        "Some-college": "College",
        "Assoc-acdm": "College",
        "Assoc-voc": "College",

        "Bachelors": "Bachelor",

        "Masters": "Postgraduate",
        "Prof-school": "Postgraduate",
        "Doctorate": "Postgraduate"
    }

    df = df.copy()

    df["EducationLevel"] = (
        df["education"]
        .map(mapping)
        .fillna("Other")
    )

    return df


# ==========================================================
# Income Per Hour
# ==========================================================

def create_income_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an approximate income-per-hour feature based on capital gain.
    """

    df = df.copy()

    df["IncomePerHour"] = np.where(
        df["hours_per_week"] > 0,
        df["capital_gain"] / df["hours_per_week"],
        0
    )

    return df


# ==========================================================
# Apply All Feature Engineering
# ==========================================================

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """

    df = create_age_groups(df)

    df = create_capital_net(df)

    df = create_work_hours_category(df)

    df = create_education_level(df)

    df = create_income_per_hour(df)

    return df


# ==========================================================
# Feature Summary
# ==========================================================

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary of engineered features.
    """

    engineered_features = [

        "AgeGroup",

        "CapitalNet",

        "WorkHoursCategory",

        "EducationLevel",

        "IncomePerHour"
    ]

    summary = pd.DataFrame({

        "Feature": engineered_features,

        "Data Type": [
            str(df[col].dtype)
            for col in engineered_features
        ],

        "Missing Values": [
            df[col].isnull().sum()
            for col in engineered_features
        ],

        "Unique Values": [
            df[col].nunique()
            for col in engineered_features
        ]
    })

    return summary
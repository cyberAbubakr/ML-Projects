"""
Preprocessing utilities for the Adult Census Income Dataset.

This module provides reusable functions for:

- Placeholder replacement
- Missing value handling
- Duplicate removal
- Category cleaning
- Feature encoding
- Feature scaling
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)


# ==========================================================
# Placeholder Values
# ==========================================================

def replace_placeholder_values(df, placeholder="?"):
    """
    Replace placeholder values with NaN.
    """
    df = df.copy()
    df.replace(placeholder, np.nan, inplace=True)
    return df


# ==========================================================
# Missing Values
# ==========================================================

def handle_missing_values(
    df,
    numerical_strategy="median",
    categorical_strategy="mode"
):
    """
    Handle missing values in numerical and categorical columns.
    """

    df = df.copy()

    numerical_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(exclude=np.number).columns

    # Numerical
    for col in numerical_columns:

        if numerical_strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())

        elif numerical_strategy == "median":
            df[col] = df[col].fillna(df[col].median())

    # Categorical
    for col in categorical_columns:

        if categorical_strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ==========================================================
# Duplicate Records
# ==========================================================

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    df = df.copy()
    return df.drop_duplicates()


# ==========================================================
# Category Cleaning
# ==========================================================

def clean_categories(df):
    """
    Remove leading/trailing spaces from categorical columns.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(exclude=np.number).columns

    for col in categorical_columns:
        df[col] = df[col].astype(str).str.strip()

    return df


# ==========================================================
# Label Encoding
# ==========================================================

def label_encode_column(df, column):
    """
    Apply Label Encoding to one column.
    """

    df = df.copy()

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    return df, encoder


# ==========================================================
# Ordinal Encoding
# ==========================================================

def ordinal_encode_columns(df, columns):
    """
    Apply Ordinal Encoding.
    """

    df = df.copy()

    encoder = OrdinalEncoder()

    df[columns] = encoder.fit_transform(df[columns])

    return df, encoder


# ==========================================================
# One-Hot Encoding
# ==========================================================

def one_hot_encode(df, columns):
    """
    Apply One-Hot Encoding.
    """

    df = pd.get_dummies(
        df,
        columns=columns,
        drop_first=True
    )

    return df


# ==========================================================
# Standard Scaling
# ==========================================================

def standard_scale(df, columns):
    """
    Standardize numerical columns.
    """

    df = df.copy()

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df, scaler


# ==========================================================
# Min-Max Scaling
# ==========================================================

def minmax_scale(df, columns):
    """
    Apply Min-Max Scaling.
    """

    df = df.copy()

    scaler = MinMaxScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df, scaler


# ==========================================================
# Helper Functions
# ==========================================================

def get_numerical_columns(df):
    """
    Return numerical columns.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    """
    Return categorical columns.
    """
    return df.select_dtypes(exclude=np.number).columns.tolist()


def dataset_summary(df):
    """
    Return preprocessing summary.
    """

    summary = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    return summary
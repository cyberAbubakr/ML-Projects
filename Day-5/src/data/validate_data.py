"""
validate_data.py

Basic data quality checks.
"""

import pandas as pd


def dataset_shape(df: pd.DataFrame):
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")


def show_columns(df: pd.DataFrame):
    print(df.columns.tolist())


def check_dtypes(df: pd.DataFrame):
    print(df.dtypes)


def missing_values(df: pd.DataFrame):
    return (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )


def duplicate_rows(df: pd.DataFrame):
    return df.duplicated().sum()


def dataset_summary(df: pd.DataFrame):

    print("=" * 60)
    dataset_shape(df)

    print("=" * 60)
    print("Duplicate Rows")
    print(duplicate_rows(df))

    print("=" * 60)
    print("Missing Values")
    print(missing_values(df))
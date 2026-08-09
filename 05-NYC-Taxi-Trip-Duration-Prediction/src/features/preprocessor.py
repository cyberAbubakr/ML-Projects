"""
preprocessor.py

Create preprocessing pipeline for the NYC Taxi
Trip Duration Prediction project.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# =============================================================================
# Create Preprocessor
# =============================================================================

def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
):
    """
    Create preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list[str]
        Numerical feature names.

    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor
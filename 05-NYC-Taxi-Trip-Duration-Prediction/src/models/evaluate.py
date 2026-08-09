"""
evaluate.py

Utilities for evaluating regression models.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

try:
    from sklearn.metrics import root_mean_squared_error

    def calculate_rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)

except ImportError:
    from sklearn.metrics import mean_squared_error

    def calculate_rmse(y_true, y_pred):
        return mean_squared_error(
            y_true,
            y_pred,
            squared=False
        )


# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Regression Evaluation Function
# =============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    training_time: float = 0.0,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained regression model.

    Parameters
    ----------
    model :
        Trained regression model.

    X_test :
        Test features.

    y_test :
        Test target values.

    training_time : float
        Training duration in seconds.

    model_name : str
        Model name.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    logger.info(
        f"Evaluating {model_name}"
    )


    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    start_time = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter() - start_time
    )


    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = calculate_rmse(
        y_test,
        predictions
    )


    median_ae = median_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    logger.info(
        f"{model_name} evaluation completed"
    )


    return {

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "Median AE": median_ae,

        "R²": r2,

        "Training Time (s)": training_time,

        "Prediction Time (s)": prediction_time,
    }



# =============================================================================
# Create Results DataFrame
# =============================================================================

def create_results_dataframe(
    results: list[dict]
) -> pd.DataFrame:
    """
    Convert evaluation results into sorted DataFrame.

    Parameters
    ----------
    results : list[dict]
        List containing model evaluation dictionaries.

    Returns
    -------
    pd.DataFrame
        Sorted results table.
    """

    df = pd.DataFrame(
        results
    )


    return (
        df
        .sort_values(
            "RMSE"
        )
        .reset_index(
            drop=True
        )
    )
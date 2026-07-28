"""
test.py

Utility functions for testing trained regression models.
"""

from __future__ import annotations


import logging
import time

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Generate Predictions
# =============================================================================

def predict_model(
    model,
    X_test,
):
    """
    Generate predictions from trained model.

    Parameters
    ----------
    model :
        Trained regression model.

    X_test :
        Processed test features.

    Returns
    -------
    predictions :
        Model predictions.
    """

    logger.info("Generating predictions...")

    start = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = time.perf_counter() - start


    logger.info(
        f"Prediction completed in {prediction_time:.3f} seconds"
    )


    return predictions, prediction_time



# =============================================================================
# Evaluate Predictions
# =============================================================================

def test_model(
    model_name: str,
    y_test,
    y_pred,
):
    """
    Calculate regression metrics.

    Parameters
    ----------
    model_name : str
        Model name.

    y_test :
        True target values.

    y_pred :
        Predicted values.

    Returns
    -------
    dict
        Evaluation results.
    """


    mae = mean_absolute_error(
        y_test,
        y_pred
    )


    rmse = mean_squared_error(
        y_test,
        y_pred,
        squared=False
    )


    r2 = r2_score(
        y_test,
        y_pred
    )


    results = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


    return results



# =============================================================================
# Compare Multiple Models
# =============================================================================

def test_multiple_models(
    models: dict,
    X_test,
    y_test,
):
    """
    Test multiple regression models.

    Parameters
    ----------
    models : dict
        Dictionary containing trained models.

    X_test :
        Processed test features.

    y_test :
        True values.

    Returns
    -------
    pd.DataFrame
        Model comparison results.
    """


    results = []


    for name, model in models.items():

        logger.info(
            f"Testing {name}"
        )


        predictions, _ = predict_model(
            model,
            X_test
        )


        result = test_model(
            name,
            y_test,
            predictions
        )


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    return results_df
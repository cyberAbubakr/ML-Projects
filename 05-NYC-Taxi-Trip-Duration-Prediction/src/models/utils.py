"""
utils.py

Utility functions for saving/loading models
and saving model evaluation results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Save Model
# =============================================================================

def save_model(
    model,
    model_name: str,
    models_dir: Path,
) -> Path:
    """
    Save a trained model.

    Parameters
    ----------
    model
        Trained model.

    model_name : str
        Name of the model.

    models_dir : Path
        Directory where models will be saved.

    Returns
    -------
    Path
        Path to the saved model.
    """

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    logger.info(f"Model saved to: {model_path}")

    return model_path


# =============================================================================
# Load Model
# =============================================================================

def load_model(model_path: Path):
    """
    Load a trained model.

    Parameters
    ----------
    model_path : Path

    Returns
    -------
    Loaded model
    """

    logger.info(f"Loading model: {model_path}")

    return joblib.load(model_path)


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    results_df: pd.DataFrame,
    reports_dir: Path,
    filename: str = "linear_results.csv",
) -> Path:
    """
    Save model comparison results.

    Parameters
    ----------
    results_df : pd.DataFrame

    reports_dir : Path

    filename : str

    Returns
    -------
    Path
    """

    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / filename

    results_df.to_csv(output_path, index=False)

    logger.info(f"Results saved to: {output_path}")

    return output_path
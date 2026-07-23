"""
Models package.

Contains utilities for training, evaluating,
saving, loading, and predicting machine learning models.
"""

from .train import (
    train_dummy_baseline,
    train_linear_regression,
    train_ridge_regression,
    train_lasso_regression,
    train_elastic_net_regression,
)

from .evaluate import evaluate_model

from .utils import (
    save_model,
    load_model,
    save_results,
)
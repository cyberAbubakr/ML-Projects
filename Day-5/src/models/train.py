"""
train.py

Training utilities for regression models.
"""

from __future__ import annotations

import logging
import time

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Training Functions
# =============================================================================

def train_dummy_baseline(X_train, y_train):
    """
    Train Dummy Median Baseline.

    Returns
    -------
    model : DummyRegressor
    training_time : float
    """

    logger.info("Training Dummy Median Baseline...")

    start = time.perf_counter()

    model = DummyRegressor(strategy="median")
    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    logger.info(f"Completed in {training_time:.3f} seconds")

    return model, training_time


def train_linear_regression(X_train, y_train):
    """
    Train Multiple Linear Regression.
    """

    logger.info("Training Linear Regression...")

    start = time.perf_counter()

    model = LinearRegression()
    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    logger.info(f"Completed in {training_time:.3f} seconds")

    return model, training_time


def train_ridge_regression(
    X_train,
    y_train,
    alpha: float = 1.0,
):
    """
    Train Ridge Regression.
    """

    logger.info(f"Training Ridge Regression (alpha={alpha})...")

    start = time.perf_counter()

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    logger.info(f"Completed in {training_time:.3f} seconds")

    return model, training_time


def train_lasso_regression(
    X_train,
    y_train,
    alpha: float = 0.001,
):
    """
    Train Lasso Regression.
    """

    logger.info(f"Training Lasso Regression (alpha={alpha})...")

    start = time.perf_counter()

    model = Lasso(
        alpha=alpha,
        random_state=42,
        max_iter=10000,
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    logger.info(f"Completed in {training_time:.3f} seconds")

    return model, training_time


def train_elastic_net_regression(
    X_train,
    y_train,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
):
    """
    Train Elastic Net Regression.
    """

    logger.info(
        f"Training Elastic Net "
        f"(alpha={alpha}, l1_ratio={l1_ratio})..."
    )

    start = time.perf_counter()

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=10000,
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    logger.info(f"Completed in {training_time:.3f} seconds")

    return model, training_time

# =============================================================================
# Decision Tree Regression
# =============================================================================

def train_decision_tree(
    X_train,
    y_train,
    max_depth=None,
):
    """
    Train Decision Tree Regressor.
    """

    logger.info("Training Decision Tree Regressor...")

    start = time.perf_counter()

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = time.perf_counter() - start

    logger.info(
        f"Completed in {training_time:.3f} seconds"
    )

    return model, training_time



# =============================================================================
# Random Forest Regression
# =============================================================================

def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
):
    """
    Train Random Forest Regressor.
    """

    logger.info("Training Random Forest Regressor...")

    start = time.perf_counter()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = time.perf_counter() - start

    logger.info(
        f"Completed in {training_time:.3f} seconds"
    )

    return model, training_time



# =============================================================================
# Gradient Boosting Regression
# =============================================================================

def train_gradient_boosting(
    X_train,
    y_train,
):
    """
    Train Gradient Boosting Regressor.
    """

    logger.info("Training Gradient Boosting Regressor...")

    start = time.perf_counter()

    model = GradientBoostingRegressor(
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = time.perf_counter() - start

    logger.info(
        f"Completed in {training_time:.3f} seconds"
    )

    return model, training_time
"""
Scikit-Learn Implementation using MLPClassifier and Hyperparameter Grid Search.
Also includes a baseline Logistic Regression to demonstrate the non-linear classification advantage of ANNs.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (16, 8),
    activation: str = "relu",
    alpha: float = 0.001,
    learning_rate_init: float = 0.01,
    max_iter: int = 500,
    random_state: int = 42
) -> MLPClassifier:
    """
    Train a standard Scikit-Learn MLPClassifier.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=20
    )
    mlp.fit(X_train, y_train)
    return mlp


def tune_mlp_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> GridSearchCV:
    """
    Perform Grid Search Cross-Validation over various ANN architectures and hyperparameter configurations.
    """
    param_grid = {
        "hidden_layer_sizes": [(8,), (16, 8), (32, 16), (32, 16, 8)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [0.01, 0.005]
    }

    base_mlp = MLPClassifier(
        solver="adam",
        max_iter=400,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )

    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def train_baseline_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LogisticRegression:
    """
    Train a linear Logistic Regression model as a baseline benchmark
    to highlight why ANNs are essential for non-linearly separable datasets.
    """
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    return lr

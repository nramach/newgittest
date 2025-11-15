"""
Simple linear regression implementation using gradient descent.

This module provides a `LinearRegression` class with optional intercept
handling, convergence tolerance, and helper utilities for evaluation.
An example usage is included in the `__main__` block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def _validate_inputs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and reshape input arrays."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim != 1:
        raise ValueError("Target vector y must be 1-dimensional.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    return X.astype(float), y.astype(float)


@dataclass
class LinearRegression:
    """Linear Regression model trained via batch gradient descent."""

    learning_rate: float = 0.01
    n_iter: int = 1000
    fit_intercept: bool = True
    tol: float = 1e-7
    random_state: Optional[int] = None

    coef_: Optional[np.ndarray] = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> "LinearRegression":
        """Fit the model to the training data."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_arr, y_arr = _validate_inputs(X_arr, y_arr)
        X_arr = self._add_intercept(X_arr)

        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(scale=0.01, size=X_arr.shape[1])

        for _ in range(self.n_iter):
            predictions = X_arr @ self.coef_
            errors = predictions - y_arr
            gradient = X_arr.T @ errors / X_arr.shape[0]

            step = self.learning_rate * gradient
            if np.linalg.norm(step) <= self.tol:
                break

            self.coef_ -= step

        return self

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict using the linear model."""
        if self.coef_ is None:
            raise RuntimeError("The model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_arr = self._add_intercept(X_arr)
        return X_arr @ self.coef_

    def score(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        y_arr = np.asarray(y, dtype=float)
        predictions = self.predict(X)
        ss_res = np.sum((y_arr - predictions) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        return 1 - ss_res / ss_tot


def mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute the mean squared error between true and predicted values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_demo = rng.uniform(-5, 5, size=(100, 1))
    noise = rng.normal(0, 1, size=100)
    y_demo = 3.5 * X_demo[:, 0] + 2.0 + noise

    model = LinearRegression(learning_rate=0.05, n_iter=5000, random_state=0)
    model.fit(X_demo, y_demo)
    predictions = model.predict(X_demo)

    print("Learned coefficients:", model.coef_)
    print("Training MSE:", mean_squared_error(y_demo, predictions))
    print("Training R^2:", model.score(X_demo, y_demo))

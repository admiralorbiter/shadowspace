"""Baseline classifier fitting for benchmark dataset probability matrix generation."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadowspace.importers.validator import validate_import_matrix


def fit_baseline_classifier(
    x_mat: NDArray[np.float64], y: NDArray[np.int64], seed: int = 20260801
) -> tuple[Any, NDArray[np.float64]]:
    """Fit a LogisticRegression classifier on x_mat, y and return (classifier, proba_matrix).

    Parameters
    ----------
    x_mat : NDArray[np.float64]
        Feature matrix of shape (N, D).
    y : NDArray[np.int64]
        Target labels of shape (N,).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[Any, NDArray[np.float64]]
        The fitted classifier model and validated probability matrix of shape (N, K).
    """
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for dataset fetching. Install via `pip install shadowspace[datasets]`."
        ) from err

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)

    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(x_scaled, y)

    raw_proba: NDArray[np.float64] = clf.predict_proba(x_scaled)
    proba_matrix = validate_import_matrix(raw_proba, normalize=True)
    return clf, proba_matrix

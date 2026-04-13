"""
probes.py

Linear probe fitting and evaluation on cached activations.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    cv: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Fit a logistic regression probe on hidden-state activations.

    Args:
        X: (n_samples, hidden_dim) float array of activations
        y: (n_samples,) int array of class labels
        C: L2 regularization strength (inverse)
        cv: Number of stratified cross-validation folds

    Returns:
        dict with keys:
            'model':       Fitted LogisticRegression (on full dataset)
            'cv_accuracy': Mean CV accuracy across folds
            'cv_auc':      Mean CV AUROC across folds
            'coef':        (n_classes, hidden_dim) weight matrix — the probe direction(s)
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes to fit a probe.")

    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )

    # Cross-validated predictions for unbiased accuracy and AUROC
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    cv_preds = cv_proba.argmax(axis=1)

    cv_accuracy = float((cv_preds == y).mean())
    # AUROC: binary case uses proba[:,1]; multiclass uses ovr
    if len(np.unique(y)) == 2:
        cv_auc = float(roc_auc_score(y, cv_proba[:, 1]))
    else:
        cv_auc = float(roc_auc_score(y, cv_proba, multi_class="ovr"))

    # Refit on full dataset for the returned model
    clf.fit(X, y)

    return {
        "model": clf,
        "cv_accuracy": cv_accuracy,
        "cv_auc": cv_auc,
        "coef": clf.coef_.copy(),
    }


def eval_probe(
    probe: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate a fitted probe on a held-out set.

    Returns:
        dict with keys: 'accuracy', 'auc', 'predictions', 'probabilities'
    """
    proba = probe.predict_proba(X)
    preds = proba.argmax(axis=1)
    accuracy = float((preds == y).mean())

    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, proba[:, 1]))
    else:
        auc = float(roc_auc_score(y, proba, multi_class="ovr"))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "predictions": preds,
        "probabilities": proba,
    }


def probe_sweep(
    activations: dict[str, np.ndarray],
    labels: np.ndarray,
    C: float = 1.0,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Fit one probe per key in activations, return a summary DataFrame.

    Args:
        activations: {module_name: (n_samples, hidden_dim)} — one entry per
                     (layer, token_position) combination
        labels: (n_samples,) int class labels

    Returns:
        DataFrame with columns [module, cv_accuracy, cv_auc]
        sorted by cv_accuracy descending.
    """
    records = []
    for module_name, X in activations.items():
        if X.shape[0] != len(labels):
            raise ValueError(
                f"Shape mismatch for '{module_name}': "
                f"X has {X.shape[0]} samples but labels has {len(labels)}."
            )
        result = fit_probe(X, labels, C=C, cv=cv)
        records.append({
            "module": module_name,
            "cv_accuracy": result["cv_accuracy"],
            "cv_auc": result["cv_auc"],
            "n_samples": len(labels),
            "hidden_dim": X.shape[1],
        })
        print(
            f"  {module_name:<60s}  "
            f"acc={result['cv_accuracy']:.3f}  auc={result['cv_auc']:.3f}"
        )

    df = pd.DataFrame(records).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
    return df


def save_probe(probe_result: dict[str, Any], path: str | Path) -> Path:
    """Save a probe result dict (including the fitted model) as a pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(probe_result, f)
    return path


def load_probe(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Permutation test for significance
# ---------------------------------------------------------------------------

def permutation_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 500,
    C: float = 1.0,
    cv: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Estimate the null accuracy distribution via label permutation.

    Returns:
        dict with keys:
            'observed_accuracy': float
            'null_accuracies':   (n_permutations,) array
            'p_value':           fraction of null >= observed
    """
    rng = np.random.default_rng(random_state)
    observed = fit_probe(X, y, C=C, cv=cv)["cv_accuracy"]

    null_accs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_accs.append(fit_probe(X, y_perm, C=C, cv=cv)["cv_accuracy"])

    null_accs = np.array(null_accs)
    p_value = float((null_accs >= observed).mean())

    return {
        "observed_accuracy": observed,
        "null_accuracies": null_accs,
        "p_value": p_value,
    }

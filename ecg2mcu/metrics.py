"""Multi-label ECG metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auc(y_true: np.ndarray, probs: np.ndarray) -> list[float]:
    """AUC per class; NaN for a class that has only positives or only negatives."""
    out = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        out.append(float(roc_auc_score(col, probs[:, c])) if 0 < col.sum() < len(col) else float("nan"))
    return out


def macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean AUC over the classes where it is defined."""
    aucs = [a for a in per_class_auc(y_true, probs) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

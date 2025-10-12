"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


@dataclass
class MetricReport:
    auc: float
    accuracy: float
    logloss: float

    def as_dict(self) -> dict[str, float]:
        return {"auc": self.auc, "accuracy": self.accuracy, "logloss": self.logloss}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> MetricReport:
    """Compute common binary classification metrics."""
    auc = roc_auc_score(y_true, y_prob)
    logloss = log_loss(y_true, y_prob, eps=1e-6)
    preds = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    return MetricReport(auc=auc, accuracy=acc, logloss=logloss)


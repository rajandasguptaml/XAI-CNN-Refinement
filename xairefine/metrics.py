"""Accuracy, macro precision / recall / F1 (Eqs. 7-10) and the confusion matrix."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> dict:
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = cm.sum(1) > 0  # macro average over classes present in the test set
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision": float(prec[present].mean()) if present.any() else 0.0,
        "recall": float(rec[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "confusion": cm.tolist(),
    }

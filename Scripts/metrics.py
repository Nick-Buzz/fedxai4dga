"""Binary-classification metrics for DGA detection.

Precision, recall and F1 are reported for the malicious (DGA) class, and
ROC-AUC / PR-AUC are computed from the model's continuous score.  Class-weighted
variants are kept under explicit ``*_weighted`` names, and operating-point
measures (recall at a fixed false-positive rate) are added because deployed DNS
traffic is overwhelmingly benign.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

POSITIVE_CLASS = 1  # 1 = DGA, 0 = benign


def binary_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Full metric set for one run.

    ``probabilities`` must be the model's continuous score, not a thresholded
    label.  Hard 0/1 labels are rejected.
    """
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)

    distinct = np.unique(probabilities)
    if distinct.size <= 2 and np.all(np.isin(distinct, (0.0, 1.0))):
        raise ValueError(
            "binary_metrics() received hard 0/1 labels as `probabilities`. "
            "ROC-AUC computed from thresholded predictions reduces to balanced "
            "accuracy. Pass model.predict(X) output before rounding."
        )

    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    result = {
        # headline
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        # positive class (DGA)
        "precision": precision_score(y_true, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall": recall_score(y_true, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1": f1_score(y_true, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        # benign side -- the operational cost
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
        # threshold-free
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        # operating points a DNS deployment is actually chosen by
        "recall_at_fpr_1pct": recall_at_fpr(y_true, probabilities, 0.01),
        "recall_at_fpr_0.1pct": recall_at_fpr(y_true, probabilities, 0.001),
        # class-weighted variants
        "precision_weighted": precision_score(y_true, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, predictions, average="weighted", zero_division=0),
        # counts
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
        "n_samples": int(tn + fp + fn + tp),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in result.items()}


def recall_at_fpr(y_true, probabilities, target_fpr: float) -> float:
    """Highest recall achievable without exceeding `target_fpr` on benign names.

    In operational DNS traffic the benign class dominates by orders of
    magnitude, so a detector is chosen by its recall at a tolerable false-alarm
    rate, not by accuracy on a 61/39 test split.
    """
    fpr, tpr, _ = roc_curve(np.asarray(y_true).reshape(-1),
                            np.asarray(probabilities).reshape(-1))
    allowed = fpr <= target_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def threshold_at_fpr(y_true, probabilities, target_fpr: float) -> float:
    """The decision threshold that realises `target_fpr`."""
    fpr, _, thresholds = roc_curve(np.asarray(y_true).reshape(-1),
                                   np.asarray(probabilities).reshape(-1))
    allowed = np.where(fpr <= target_fpr)[0]
    return float(thresholds[allowed[-1]]) if allowed.size else 1.0


def per_family_metrics(families, y_true, probabilities, threshold: float = 0.5):
    """Recall per DGA family, and false-positive rate on the benign rows.

    Reported per family because a mean over 38 families hides the ones the
    detector cannot see at all -- which is exactly what a security reader wants
    to know.
    """
    import pandas as pd

    frame = pd.DataFrame({
        "family": np.asarray(families).reshape(-1),
        "y_true": np.asarray(y_true).reshape(-1),
        "probability": np.asarray(probabilities).reshape(-1),
    })
    frame["prediction"] = (frame["probability"] >= threshold).astype(int)

    rows = []
    for family, group in frame.groupby("family", sort=True):
        truth = group["y_true"].to_numpy()
        pred = group["prediction"].to_numpy()
        positive = truth.sum()
        rows.append({
            "family": family,
            "samples": len(group),
            "detected": int(pred.sum()) if positive else None,
            "recall": float(pred.sum() / positive) if positive else None,
            "false_positive_rate": float(pred.sum() / len(group)) if not positive else None,
            "mean_probability": float(group["probability"].mean()),
        })
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)


def aggregate(per_seed_metrics, keys=None):
    """Mean / std / min / max across seeds, for the results table."""
    import pandas as pd

    frame = pd.DataFrame(list(per_seed_metrics))
    if keys is None:
        keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "specificity", "false_positive_rate", "roc_auc", "pr_auc", "mcc",
                "recall_at_fpr_1pct"]
    keys = [k for k in keys if k in frame.columns]
    return pd.DataFrame({
        "metric": keys,
        "mean": [frame[k].mean() for k in keys],
        "std": [frame[k].std(ddof=1) for k in keys],
        "min": [frame[k].min() for k in keys],
        "max": [frame[k].max() for k in keys],
    })

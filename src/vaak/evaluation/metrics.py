import numpy as np
from sklearn.metrics import det_curve, roc_auc_score


def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    pos_label: int = 1,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and its threshold."""
    fpr, fnr, thresholds = det_curve(
        y_true=labels,
        y_score=scores,
        pos_label=pos_label,
    )

    idx = int(np.nanargmin(np.absolute(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return eer, threshold


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    pos_label: int = 1,
) -> float:
    """Compute normalized minimum Detection Cost Function (minDCF)."""
    fpr, fnr, _ = det_curve(
        y_true=labels,
        y_score=scores,
        pos_label=pos_label,
    )

    costs = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    min_cost = np.min(costs)
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))

    return float(min_cost / default_cost)


def compute_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute Receiver Operating Characteristic Area Under the Curve (ROC-AUC)."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    return float(roc_auc_score(y_true=labels, y_score=scores))

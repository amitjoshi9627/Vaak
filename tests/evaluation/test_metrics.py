import numpy as np

from vaak.evaluation.metrics import compute_auc, compute_eer, compute_min_dcf


def test_compute_eer() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    eer, threshold = compute_eer(labels, scores)

    assert 0.0 <= eer <= 1.0
    assert 0.1 <= threshold <= 0.9


def test_compute_min_dcf() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    min_dcf = compute_min_dcf(labels, scores)

    assert min_dcf >= 0.0


def test_compute_auc() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    auc = compute_auc(labels, scores)

    assert auc == 1.0

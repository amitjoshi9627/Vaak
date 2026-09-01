from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from vaak.data.dataset import VaakBatch


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and operating threshold."""
    if len(np.unique(y_true)) < 2:
        return float("inf"), 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    diffs = np.absolute(fnr - fpr)

    if np.all(np.isnan(diffs)):
        return float("inf"), 0.5

    eer_index = int(np.nanargmin(diffs))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    threshold = float(thresholds[eer_index])
    return eer, threshold


def compute_normalized_min_dcf(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute normalized minimum Detection Cost Function (minDCF).

    Note: This is a pairwise cost benchmark metric with p_target=0.05, distinct from
    the SASV/ASVspoof tandem t-DCF metric.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    c_det = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = float(np.min(c_det) / c_default)
    return min_dcf


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation summary with utterance-level scoring."""

    eer: float
    threshold: float
    auc: float
    norm_min_dcf: float
    attack_metrics: dict[str, dict[str, float]]
    total_samples: int


class Evaluator:
    """Evaluates detector across entire recordings using multi-chunk aggregation."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable[VaakBatch]) -> EvaluationReport:
        """Run full-utterance evaluation over dataloader (batch_size=1)."""
        self.model.eval()

        utterance_scores: list[float] = []
        labels: list[int] = []
        attack_ids: list[str | None] = []

        for batch in dataloader:
            # batch['audio'] is [1, num_chunks, 64000] -> squeeze to [num_chunks, 64000]
            chunks = batch["audio"].squeeze(0).to(self.device)

            # The collate function renames these to plural 'labels' and 'attack_ids'
            label = int(batch["labels"][0].item())
            atk_id = batch["attack_ids"][0]

            # Predict across all chunks in the recording: [K, 2]
            logits = cast(torch.Tensor, self.model(chunks))

            # Extract Spoof probability (positive class = index 1)
            chunk_spoof_probs = torch.softmax(logits, dim=1)[:, 1]

            # Utterance score is mean spoof probability across all chunks
            utterance_score = float(chunk_spoof_probs.mean().item())

            utterance_scores.append(utterance_score)
            labels.append(label)
            attack_ids.append(atk_id)

        y_true = np.array(labels, dtype=int)
        y_scores = np.array(utterance_scores, dtype=float)

        if len(np.unique(y_true)) >= 2:
            global_eer, global_threshold = compute_eer(y_true, y_scores)
            global_auc = float(roc_auc_score(y_true, y_scores))
            global_min_dcf = compute_normalized_min_dcf(y_true, y_scores)
        else:
            global_eer, global_threshold = float("inf"), 0.5
            global_auc = 0.0
            global_min_dcf = float("inf")

        # Disaggregated pairwise evaluation: Bona Fide vs. Attack Axx
        attack_metrics: dict[str, dict[str, float]] = {}
        unique_attacks = {atk for atk in attack_ids if atk is not None}
        bona_fide_mask = y_true == 0

        for atk in sorted(unique_attacks):
            attack_mask = np.array([atk_id == atk for atk_id in attack_ids])
            subset_mask = bona_fide_mask | attack_mask

            if np.sum(attack_mask) > 0 and np.sum(bona_fide_mask) > 0:
                sub_y_true = y_true[subset_mask]
                sub_y_scores = y_scores[subset_mask]

                sub_eer, _ = compute_eer(sub_y_true, sub_y_scores)
                sub_auc = float(roc_auc_score(sub_y_true, sub_y_scores))
                sub_dcf = compute_normalized_min_dcf(sub_y_true, sub_y_scores)
            else:
                sub_eer = float("inf")
                sub_auc = 0.0
                sub_dcf = float("inf")

            attack_metrics[atk] = {
                "attack_vs_bonafide_eer": sub_eer,
                "attack_vs_bonafide_auc": sub_auc,
                "attack_vs_bonafide_norm_min_dcf": sub_dcf,
            }

        return EvaluationReport(
            eer=global_eer,
            threshold=global_threshold,
            auc=global_auc,
            norm_min_dcf=global_min_dcf,
            attack_metrics=attack_metrics,
            total_samples=len(y_true),
        )

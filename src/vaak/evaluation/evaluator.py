from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vaak.core.logging import get_logger
from vaak.data.dataset import VaakBatch
from vaak.evaluation.metrics import compute_auc, compute_eer, compute_min_dcf

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    """Comprehensive evaluation metrics report."""

    eer: float
    min_dcf: float
    auc: float
    attack_metrics: dict[str, dict[str, float]]


class Evaluator:
    """Standard evaluation harness for Vaak models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader[VaakBatch],
    ) -> EvaluationReport:
        """Run inference over a dataset and compute detailed metrics."""
        logger.info(f"Running evaluation on {self.device}")

        all_labels: list[int] = []
        all_scores: list[float] = []
        all_attack_ids: list[str | None] = []

        for batch in dataloader:
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].cpu().tolist()
            attack_ids = batch["attack_ids"]

            logits = cast(torch.Tensor, self.model(audio))

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            all_labels.extend(labels)
            all_scores.extend(probs)
            all_attack_ids.extend(attack_ids)

        labels_arr = np.array(all_labels)
        scores_arr = np.array(all_scores)
        attack_ids_arr = np.array(all_attack_ids)

        overall_eer, _ = compute_eer(labels_arr, scores_arr)
        overall_min_dcf = compute_min_dcf(labels_arr, scores_arr)
        overall_auc = compute_auc(labels_arr, scores_arr)

        logger.info(
            f"Overall Metrics - EER: {overall_eer:.4f} | "
            f"minDCF: {overall_min_dcf:.4f} | "
            f"AUC: {overall_auc:.4f}"
        )

        attack_metrics = self._compute_attack_metrics(
            labels=labels_arr,
            scores=scores_arr,
            attack_ids=attack_ids_arr,
        )

        return EvaluationReport(
            eer=overall_eer,
            min_dcf=overall_min_dcf,
            auc=overall_auc,
            attack_metrics=attack_metrics,
        )

    def _compute_attack_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        attack_ids: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute metrics disaggregated by attack identifier."""
        attack_metrics: dict[str, dict[str, float]] = {}

        bonafide_mask = labels == 0
        bonafide_labels = labels[bonafide_mask]
        bonafide_scores = scores[bonafide_mask]

        unique_attacks = set(attack_ids) - {None}

        for attack_id in sorted(unique_attacks):
            attack_mask = attack_ids == attack_id
            attack_labels = labels[attack_mask]
            attack_scores = scores[attack_mask]

            if len(attack_labels) == 0:
                continue

            slice_labels = np.concatenate([bonafide_labels, attack_labels])
            slice_scores = np.concatenate([bonafide_scores, attack_scores])

            eer, _ = compute_eer(slice_labels, slice_scores)
            min_dcf = compute_min_dcf(slice_labels, slice_scores)
            auc = compute_auc(slice_labels, slice_scores)

            attack_metrics[str(attack_id)] = {
                "eer": eer,
                "min_dcf": min_dcf,
                "auc": auc,
            }

        return attack_metrics

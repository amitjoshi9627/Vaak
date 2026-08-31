import torch
import torch.nn as nn

from vaak.data.dataset import VaakBatch
from vaak.evaluation.evaluator import Evaluator


class DummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [K, 64000] (K chunks per utterance) -> return [K, 2]
        num_chunks = x.shape[0]
        return torch.randn(num_chunks, 2)


def test_evaluator_utterance_aggregation() -> None:
    model = DummyModel()
    evaluator = Evaluator(model=model, device=torch.device("cpu"))

    # Utterance 1: Bona Fide (label 0, attack None)
    batch_bonafide: VaakBatch = {
        "audio": torch.randn(1, 4, 64000),
        "labels": torch.tensor([0]),
        "sample_ids": ["s1"],
        "speaker_ids": ["spk1"],
        "datasets": ["asv2019"],
        "attack_ids": [None],
    }

    # Utterance 2: Spoof (label 1, attack A01)
    batch_spoof: VaakBatch = {
        "audio": torch.randn(1, 4, 64000),
        "labels": torch.tensor([1]),
        "sample_ids": ["s2"],
        "speaker_ids": ["spk2"],
        "datasets": ["asv2019"],
        "attack_ids": ["A01"],
    }

    # Pass as an iterable of batches
    report = evaluator.evaluate([batch_bonafide, batch_spoof])

    assert hasattr(report, "norm_min_dcf")
    assert hasattr(report, "eer")
    assert report.total_samples == 2

    assert "A01" in report.attack_metrics
    attack_stats = report.attack_metrics["A01"]
    assert "attack_vs_bonafide_eer" in attack_stats
    assert "attack_vs_bonafide_norm_min_dcf" in attack_stats

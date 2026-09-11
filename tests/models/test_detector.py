import torch
import torch.nn as nn

from vaak.models.detector import VaakDetector


class DummyEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates returning 13 layers: [13, B, T, D]
        batch_size = x.size(0)
        return torch.randn(13, batch_size, 50, 768)


class DummyAggregator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates fusion: [13, B, T, D] -> [B, T, D]
        return x.mean(dim=0)


class DummyTemporal(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates sequence modeling (CNN/Conformer): [B, T, D] -> [B, T, D]
        return x


class DummyPooler(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates ASP pooling: [B, T, D] -> [B, 1536]
        batch_size = x.size(0)
        return torch.randn(batch_size, 1536)


class DummyHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates linear projection: [B, 1536] -> [B, 2]
        batch_size = x.size(0)
        return torch.zeros(batch_size, 2)


def test_vaak_detector_forward_pass() -> None:
    batch_size = 4
    # Dummy raw audio input [B, L]
    audio = torch.randn(batch_size, 64000)

    detector = VaakDetector(
        encoder=DummyEncoder(),
        aggregator=DummyAggregator(),
        temporal=DummyTemporal(),
        pooler=DummyPooler(),
        head=DummyHead(),
    )

    logits = detector(audio)

    # The final output must be classification logits for real vs spoof
    assert logits.shape == (batch_size, 2)

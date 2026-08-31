from typing import cast

import torch
import torch.nn as nn


class VaakDetector(nn.Module):
    """Composed anti-spoofing detector pipeline."""

    def __init__(
        self,
        encoder: nn.Module,
        backend: nn.Module,
        head: nn.Module,
        aggregator: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.backend = backend
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        if self.aggregator is not None:
            features = self.aggregator(features)

        pooled = self.backend(features)
        logits = self.head(pooled)
        return cast(torch.Tensor, logits)

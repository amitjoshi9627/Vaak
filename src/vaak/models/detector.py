from typing import cast

import torch
import torch.nn as nn


class VaakDetector(nn.Module):
    """Composed spoofing detection model."""

    def __init__(
        self,
        encoder: nn.Module,
        backend: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backend = backend
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a full forward pass.

        Args:
            x: Audio tensor of shape [batch_size, sequence_length].

        Returns:
            Logits of shape [batch_size, 2].
        """
        features = cast(torch.Tensor, self.encoder(x))
        pooled = cast(torch.Tensor, self.backend(features))
        logits = cast(torch.Tensor, self.head(pooled))

        return logits

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Temporal mean pooling backend."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features over the sequence dimension.

        Args:
            x: Tensor of shape [batch_size, sequence_length, hidden_size].

        Returns:
            Tensor of shape [batch_size, hidden_size].
        """
        return x.mean(dim=1)

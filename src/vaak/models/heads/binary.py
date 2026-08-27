from typing import cast

import torch
import torch.nn as nn


class BinaryLinearHead(nn.Module):
    """Binary classification head for spoof detection."""

    def __init__(
        self,
        input_dim: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute binary classification logits.

        Args:
            x: Tensor of shape [batch_size, input_dim].

        Returns:
            Logits of shape [batch_size, 2].
        """
        x = self.dropout(x)

        return cast(torch.Tensor, self.linear(x))

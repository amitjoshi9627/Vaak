from typing import cast

import torch
import torch.nn as nn


class FrameProjection(nn.Module):
    """A frame-level bottleneck to compress features before temporal pooling."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
        Returns:
            Projected tensor of shape [B, T, output_dim]
        """
        return cast(torch.Tensor, self.drop(self.act(self.norm(self.proj(x)))))

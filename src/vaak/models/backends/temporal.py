from typing import cast

import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """1D Convolutional sequence model to capture deepfake artifacts over time."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_channels = hidden_channels or channels

        # padding=kernel_size//2 ensures the time dimension length stays exactly the same
        self.conv_net = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                hidden_channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(channels),
            nn.Dropout(p=dropout),
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, D]
        Returns:
            Sequence-aware tensor of shape [B, T, D]
        """
        # Conv 1d expects [Batch, Channels, Time]
        x_transposed = x.transpose(1, 2)

        # Pass through CNN
        conv_out = self.conv_net(x_transposed)

        # Residual connection + activation
        out_transposed = self.act(x_transposed + conv_out)

        return cast(torch.Tensor, out_transposed.transpose(1, 2))

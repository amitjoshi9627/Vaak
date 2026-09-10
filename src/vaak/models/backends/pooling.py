import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    """Original basic mean pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class AttentiveStatisticsPooling(nn.Module):
    """Calculates attention-weighted mean and standard deviation over time."""

    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden representations of shape [B, T, D].

        Returns:
            Pooled representation of shape [B, 2 * D].
        """
        # Calculate unnormalized attention scores: [B, T, 1]
        attn_scores = self.attention(x)

        # Normalize across the time dimension: [B, T, 1]
        alpha = F.softmax(attn_scores, dim=1)

        # (B, T, D) * (B, T, 1) -> sum over T -> (B, D)
        weighted_mean = (x * alpha).sum(dim=1)

        # Clamp variance to a tiny positive number to avoid NaN in sqrt
        variance = (x - weighted_mean.unsqueeze(1)).pow(2)
        weighted_variance = (variance * alpha).sum(dim=1)
        weighted_std = torch.sqrt(torch.clamp(weighted_variance, min=1e-8))

        # Concatenate mean and std along the feature dimension
        return torch.cat([weighted_mean, weighted_std], dim=1)

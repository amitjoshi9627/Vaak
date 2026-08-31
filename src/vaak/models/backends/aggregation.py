import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLayerAggregation(nn.Module):
    """Learnable weighted sum of stacked hidden layers that adapts dynamically."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, stacked_states: torch.Tensor) -> torch.Tensor:
        """Aggregate layers using a learned Softmax distribution.

        Args:
            stacked_states: Tensor of shape [num_layers, B, T, D].

        Returns:
            Fused temporal representation of shape [B, T, D].
        """
        num_layers = stacked_states.size(0)

        if num_layers != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} layers, but received {num_layers}."
            )

        alpha = F.softmax(self.weights, dim=0)

        alpha = alpha.view(num_layers, 1, 1, 1)

        # Multiply and sum along the num_layers dimension (dim=0)
        return (stacked_states * alpha).sum(dim=0)

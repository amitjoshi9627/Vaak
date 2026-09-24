import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineHead(nn.Module):
    """
    Project features onto a unit hypersphere for metric learning.
    Outputs scaled cosine similarities.
    """

    def __init__(
        self, input_dim: int, num_classes: int = 2, scale: float = 30.0
    ) -> None:
        super().__init__()
        self.scale = scale
        # The learnable class centers (anchors)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the incoming features and the weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Calculate cosine similarity [-1, 1] and scale it up for softmax
        # Shape: [Batch, num_classes]
        return self.scale * F.linear(x_norm, w_norm)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """
    Corrected Focal Loss for imbalanced multi-class classification tasks.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # alpha should be a tensor of shape (num_classes,) containing weights per class
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute standard Cross-Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute pt (probability of the true class)
        pt = torch.exp(-ce_loss)

        # Compute the modulating factor (1 - pt)^gamma
        # Clamping avoids minor numerical instabilities when pt is exactly 1 or 0
        modulating_factor = (1.0 - pt).clamp(min=0.0, max=1.0) ** self.gamma
        focal_loss = modulating_factor * ce_loss

        # Correctly apply class-specific alpha weights if provided
        if self.alpha is not None:
            # Gather the specific alpha value for each target token/pixel/sample
            # Ensure alpha is on the correct device
            self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]
            focal_loss = at * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss

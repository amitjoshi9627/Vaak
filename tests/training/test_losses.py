import torch
import torch.nn.functional as F

from vaak.training.losses import MultiClassFocalLoss


def test_focal_loss_forward_pass() -> None:
    batch_size = 4
    num_classes = 2

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 0, 1])

    criterion = MultiClassFocalLoss(gamma=2.0)
    loss = criterion(inputs, targets)

    assert loss.dim() == 0  # Should be a scalar due to reduction="mean"
    assert loss.item() >= 0.0


def test_focal_loss_with_alpha() -> None:
    batch_size = 4
    num_classes = 2

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 0, 1])
    alpha = torch.tensor([0.25, 0.75])

    criterion = MultiClassFocalLoss(alpha=alpha, gamma=2.0)
    loss = criterion(inputs, targets)

    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_focal_loss_gamma_zero_equals_ce() -> None:
    batch_size = 4
    num_classes = 3

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 2, 1, 0])

    focal_criterion = MultiClassFocalLoss(gamma=0.0)
    focal_loss = focal_criterion(inputs, targets)

    ce_loss = F.cross_entropy(inputs, targets, reduction="mean")

    # When gamma=0 and alpha=None, focal loss should be identical to CE loss
    assert torch.allclose(focal_loss, ce_loss)

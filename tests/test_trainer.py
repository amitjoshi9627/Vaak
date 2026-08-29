from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from vaak.data.dataset import VaakBatch
from vaak.models.backends.pooling import MeanPooling
from vaak.models.detector import VaakDetector
from vaak.models.heads.binary import BinaryLinearHead
from vaak.training.trainer import Trainer


class DummyEncoder(nn.Module):
    """Lightweight dummy encoder for testing without downloading full weights."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, hidden_dim, kernel_size=100, stride=50)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for dummy encoder."""
        out = cast(torch.Tensor, self.conv(x.unsqueeze(1)))
        return out.transpose(1, 2)


def test_trainer_fit_loop(tmp_path: Path) -> None:
    device = torch.device("cpu")
    encoder = DummyEncoder(hidden_dim=32)
    backend = MeanPooling()
    head = BinaryLinearHead(input_dim=32)
    model = VaakDetector(encoder=encoder, backend=backend, head=head)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_batch: VaakBatch = {
        "audio": torch.randn(4, 64_000),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "sample_ids": ["s1", "s2", "s3", "s4"],
        "speaker_ids": ["spk1", "spk1", "spk2", "spk2"],
        "datasets": ["test", "test", "test", "test"],
        "attack_ids": [None, "A01", None, "A02"],
    }

    train_loader = [dummy_batch]
    val_loader = [dummy_batch]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=tmp_path,
    )

    result = trainer.fit(
        train_loader=train_loader,  # type: ignore[arg-type]
        val_loader=val_loader,  # type: ignore[arg-type]
        epochs=2,
    )

    assert len(result.train_losses) == 2
    assert len(result.val_losses) == 2
    assert (tmp_path / "best_model.pt").exists()
    assert result.total_training_time_seconds > 0.0

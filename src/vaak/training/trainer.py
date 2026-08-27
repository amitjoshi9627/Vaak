from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vaak.core.logging import get_logger
from vaak.data.dataset import VaakBatch

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Summary of training metrics across epochs."""

    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]


class Trainer:
    """Standard training and evaluation manager for Vaak models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader[VaakBatch]) -> float:
        """Run a single training epoch.

        Args:
            dataloader: DataLoader supplying VaakBatch samples.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = cast(torch.Tensor, self.model(audio))
            loss = cast(torch.Tensor, self.criterion(logits, labels))

            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

            batch_size = audio.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader[VaakBatch]) -> tuple[float, float]:
        """Run validation evaluation.

        Args:
            dataloader: DataLoader supplying VaakBatch samples.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataloader:
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = cast(torch.Tensor, self.model(audio))
            loss = cast(torch.Tensor, self.criterion(logits, labels))

            batch_size = audio.size(0)
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

        average_loss = running_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return average_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader[VaakBatch],
        val_loader: DataLoader[VaakBatch],
        epochs: int,
    ) -> TrainingResult:
        """Execute full training loop over multiple epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Total number of epochs to train.

        Returns:
            TrainingResult containing loss trajectories and best validation loss.
        """
        best_val_loss = float("inf")
        train_losses: list[float] = []
        val_losses: list[float] = []

        logger.info(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.checkpoint_dir is not None:
                    self._save_checkpoint("best_model.pt", epoch, val_loss)

        return TrainingResult(
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
        )

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model state dictionary and metadata.

        Args:
            filename: Destination file name.
            epoch: Current epoch number.
            val_loss: Validation loss associated with checkpoint.
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

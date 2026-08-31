import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vaak.core.logging import get_logger
from vaak.data.dataset import VaakBatch
from vaak.utils.tracker import MLflowTracker

logger = get_logger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a human-readable string."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes > 0:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.2f}s"


@dataclass(frozen=True)
class TrainingResult:
    """Summary of training metrics across epochs."""

    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    total_training_time_seconds: float
    epoch_durations_seconds: list[float]


class Trainer:
    """Standard training and evaluation manager for Vaak models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path | None = None,
        log_interval: int = 100,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.tracker = tracker

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader[VaakBatch], epoch: int) -> float:
        """Run a single training epoch and log intermediate progress.

        Args:
            dataloader: DataLoader supplying VaakBatch samples.
            epoch: The current epoch index (for logging purposes).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader, start=1):
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

            if batch_idx % self.log_interval == 0 or batch_idx == num_batches:
                current_loss = running_loss / total_samples
                logger.info(
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Loss: {current_loss:.4f}"
                )

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

            if audio.ndim == 2:
                # Standard batch:
                # [B, L]
                logits = cast(torch.Tensor, self.model(audio))

            elif audio.ndim == 3:
                # Utterance-level evaluation:
                # [1, K, L]
                if audio.size(0) != 1:
                    raise ValueError(
                        "EvaluationDataset requires batch_size=1. "
                        f"Got batch size {audio.size(0)}."
                    )

                # Remove the DataLoader batch dimension.
                # [1, K, L] -> [K, L]
                chunks = audio.squeeze(0)

                # Run the model independently on every chunk.
                # [K, L] -> [K, 2]
                chunk_logits = cast(torch.Tensor, self.model(chunks))

                # Aggregate chunk logits into ONE utterance prediction.
                # [K, 2] -> [1, 2]
                logits = chunk_logits.mean(
                    dim=0,
                    keepdim=True,
                )

            else:
                raise ValueError(
                    "Expected audio with shape [B, L] or [1, K, L], "
                    f"got {tuple(audio.shape)}."
                )

            loss = cast(
                torch.Tensor,
                self.criterion(logits, labels),
            )

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

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
        epoch_durations: list[float] = []

        logger.info(f"Starting training on device: {self.device}")
        total_start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.perf_counter()

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)

            epoch_duration = time.perf_counter() - epoch_start_time
            epoch_durations.append(epoch_duration)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch} Summary | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Duration: {format_duration(epoch_duration)}"
            )

            if self.tracker is not None:
                self.tracker.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "epoch_duration_seconds": epoch_duration,
                    },
                    step=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.checkpoint_dir is not None:
                    self._save_checkpoint("best_model.pt", epoch, val_loss)

        total_duration = time.perf_counter() - total_start_time
        logger.info(
            f"Training Complete | Total Time: {format_duration(total_duration)} | "
            f"Avg Epoch Time: {format_duration(total_duration / epochs)}"
        )

        if self.tracker is not None:
            self.tracker.log_metrics({"total_training_time_seconds": total_duration})

        return TrainingResult(
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            total_training_time_seconds=total_duration,
            epoch_durations_seconds=epoch_durations,
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

        if self.tracker is not None:
            self.tracker.log_artifact(checkpoint_path)

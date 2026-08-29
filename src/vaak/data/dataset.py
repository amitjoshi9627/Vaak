import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset

from vaak.audio import AudioPipeline

logger = logging.getLogger(__name__)


class VaakSample(TypedDict):
    """A single training/evaluation sample."""

    audio: torch.Tensor
    label: int
    sample_id: str
    speaker_id: str
    dataset: str
    attack_id: str | None


class VaakBatch(TypedDict):
    """A batch of Vaak samples."""

    audio: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]
    speaker_ids: list[str]
    datasets: list[str]
    attack_ids: list[str | None]


class VaakDataset(Dataset[VaakSample]):
    """PyTorch dataset backed by a Vaak manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        audio_pipeline: AudioPipeline,
        project_root: Path,
        random_crop: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.manifest = manifest.loc[manifest["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            logger.info(
                f"Limiting {split} samples to {max_samples} samples out of {len(self.manifest)}"
            )
            self.manifest = self.manifest.head(max_samples)

        if self.manifest.empty:
            raise ValueError(f"No samples found for split '{split}'.")

        self.audio_pipeline = audio_pipeline
        self.project_root = project_root
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> VaakSample:
        row = self.manifest.iloc[index]

        audio_path = self.project_root / str(row["audio_path"])

        chunks = self.audio_pipeline.process(audio_path)

        audio = self._random_crop(chunks) if self.random_crop else chunks[0]

        attack_id = row["attack_id"]

        return {
            "audio": audio,
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "dataset": str(row["dataset"]),
            "attack_id": (None if pd.isna(attack_id) else str(attack_id)),
        }

    @staticmethod
    def _random_crop(chunks: torch.Tensor) -> torch.Tensor:
        """Select one randomly sampled chunk."""

        if chunks.ndim != 2:
            raise ValueError("Expected chunks with shape [num_chunks, samples].")

        index = int(
            torch.randint(
                low=0,
                high=chunks.shape[0],
                size=(1,),
            ).item()
        )

        return chunks[index]


def vaak_collate_fn(
    samples: list[VaakSample],
) -> VaakBatch:
    """Collate Vaak samples into a training batch."""

    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    return {
        "audio": torch.stack([sample["audio"] for sample in samples]),
        "labels": torch.tensor(
            [sample["label"] for sample in samples],
            dtype=torch.long,
        ),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "speaker_ids": [sample["speaker_id"] for sample in samples],
        "datasets": [sample["dataset"] for sample in samples],
        "attack_ids": [sample["attack_id"] for sample in samples],
    }

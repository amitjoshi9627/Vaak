import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset

from vaak.audio.pipeline import AudioPipeline

logger = logging.getLogger(__name__)


class VaakSample(TypedDict):
    """Single dataset sample.

    Training:
        audio has shape [L].

    Evaluation:
        audio has shape [K, L], where K is the number of
        overlapping chunks for the utterance.
    """

    audio: torch.Tensor
    label: int
    sample_id: str
    speaker_id: str
    dataset: str
    attack_id: str | None


class VaakBatch(TypedDict):
    """Batch produced by the training/evaluation collator.

    Training batches contain:
        audio -> [B, L]

    Evaluation with batch_size=1 contains:
        audio -> [1, K, L]
    """

    audio: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]
    speaker_ids: list[str]
    datasets: list[str]
    attack_ids: list[str | None]


class BaseVaakDataset(Dataset[VaakSample], ABC):
    """Base dataset backed by a Vaak manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        audio_pipeline: AudioPipeline,
        project_root: Path,
        max_samples: int | None = None,
    ) -> None:
        self.manifest = manifest.loc[manifest["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            logger.warning(
                "Limiting %s split to %d samples (non-benchmark/debug mode).",
                split,
                max_samples,
            )
            self.manifest = self.manifest.sample(
                n=max_samples,
                random_state=42,
            )

        if self.manifest.empty:
            raise ValueError(f"No samples found for split '{split}'.")

        self.audio_pipeline = audio_pipeline
        self.project_root = project_root

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.manifest)

    @abstractmethod
    def __getitem__(self, index: int) -> VaakSample:
        """Return one dataset sample."""
        raise NotImplementedError


def vaak_collate_fn(
    samples: list[VaakSample],
) -> VaakBatch:
    """Collate samples into a batch.

    This collator is safe for both:
    - training samples: [L]
    - evaluation samples when batch_size=1: [K, L]

    Evaluation should use batch_size=1 because different utterances
    may contain different numbers of chunks.
    """
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

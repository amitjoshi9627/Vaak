from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from vaak.audio import AudioPipeline


class VaakAudioDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset for Vaak audio classification."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        audio_pipeline: AudioPipeline,
    ) -> None:
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        if self.manifest.empty:
            raise ValueError(f"No samples found for split: {split}")

        self.audio_pipeline = audio_pipeline

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:

        row = self.manifest.iloc[index]

        audio_path = Path(row["audio_path"])
        label = int(row["label"])

        chunks = self.audio_pipeline.process(audio_path)

        return chunks, label

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from vaak.audio import AudioPipeline


class VaakAudioDataset(Dataset[dict[str, object]]):
    """Dataset for Vaak audio samples."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        audio_pipeline: AudioPipeline,
    ) -> None:
        self.manifest = manifest.loc[manifest["split"] == split].reset_index(drop=True)

        if self.manifest.empty:
            raise ValueError(f"No samples found for split '{split}'.")

        self.audio_pipeline = audio_pipeline

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]

        chunks = self.audio_pipeline.process(Path(row["audio_path"]))

        return {
            "audio": chunks,
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "dataset": str(row["dataset"]),
            "attack_id": (None if pd.isna(row["attack_id"]) else str(row["attack_id"])),
        }

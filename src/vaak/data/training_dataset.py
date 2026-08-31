import pandas as pd
import torch

from vaak.data.dataset import BaseVaakDataset, VaakSample


class TrainingDataset(BaseVaakDataset):
    """Dataset that returns one random fixed-length chunk per utterance."""

    def __getitem__(self, index: int) -> VaakSample:
        row = self.manifest.iloc[index]

        audio_path = self.project_root / str(row["audio_path"])

        chunks = self.audio_pipeline.process(audio_path)

        if chunks.ndim != 2:
            raise ValueError(
                "Expected chunks with shape "
                f"[num_chunks, samples], got {tuple(chunks.shape)}."
            )

        if chunks.shape[0] == 0:
            raise ValueError(f"No audio chunks produced for {audio_path}.")

        chunk_index = int(
            torch.randint(
                low=0,
                high=chunks.shape[0],
                size=(1,),
            ).item()
        )

        audio = chunks[chunk_index]

        attack_id = row["attack_id"]

        return {
            "audio": audio,
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "dataset": str(row["dataset"]),
            "attack_id": (None if pd.isna(attack_id) else str(attack_id)),
        }

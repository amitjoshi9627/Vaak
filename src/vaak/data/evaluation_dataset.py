import pandas as pd

from vaak.data.dataset import BaseVaakDataset, VaakSample


class EvaluationDataset(BaseVaakDataset):
    """Dataset that returns all deterministic chunks for one utterance."""

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

        attack_id = row["attack_id"]

        return {
            "audio": chunks,
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "dataset": str(row["dataset"]),
            "attack_id": (None if pd.isna(attack_id) else str(attack_id)),
        }

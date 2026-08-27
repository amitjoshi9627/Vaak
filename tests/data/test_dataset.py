from pathlib import Path

import pandas as pd
import torch

from vaak.audio import AudioPipeline, AudioPipelineConfig
from vaak.data.dataset import VaakAudioDataset

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "manifests" / "test_manifest.csv"
)


def build_dataset() -> VaakAudioDataset:
    """Build a demo Vaak dataset from the test manifest."""

    manifest = pd.read_csv(MANIFEST_PATH)

    project_root = Path(__file__).resolve().parents[2]

    manifest["audio_path"] = manifest["audio_path"].map(
        lambda path: str(project_root / path)
    )

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            sample_rate=16_000,
            chunk_duration_seconds=4.0,
            hop_duration_seconds=2.0,
        )
    )

    return VaakAudioDataset(
        manifest=manifest,
        split="train",
        audio_pipeline=pipeline,
    )


def test_dataset_length() -> None:
    """Dataset length should equal the selected split."""

    dataset = build_dataset()

    assert len(dataset) == 2


def test_dataset_returns_expected_structure() -> None:
    """A dataset item should contain audio and metadata."""

    dataset = build_dataset()

    sample = dataset[0]

    assert set(sample) == {
        "audio",
        "label",
        "sample_id",
        "speaker_id",
        "dataset",
        "attack_id",
    }


def test_dataset_returns_audio_tensor() -> None:
    """Audio should come from the shared AudioPipeline."""

    dataset = build_dataset()

    sample = dataset[0]

    audio = sample["audio"]

    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 2
    assert audio.shape[1] == 64_000


def test_dataset_returns_label() -> None:
    """Labels should be integers."""

    dataset = build_dataset()

    sample = dataset[0]

    assert sample["label"] == 0


def test_dataset_preserves_metadata() -> None:
    """Dataset metadata should survive loading."""

    dataset = build_dataset()

    sample = dataset[0]

    assert sample["sample_id"] == "sample_001"
    assert sample["speaker_id"] == "spk_001"
    assert sample["dataset"] == "demo"

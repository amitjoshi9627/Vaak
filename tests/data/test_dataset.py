from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from vaak.audio import AudioPipeline, AudioPipelineConfig
from vaak.data.dataset import VaakDataset, vaak_collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "manifests" / "test_manifest.csv"


def build_dataset(
    split: str,
    random_crop: bool = False,
) -> VaakDataset:
    manifest = pd.read_csv(MANIFEST_PATH)

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            sample_rate=16_000,
            chunk_duration_seconds=4.0,
            hop_duration_seconds=2.0,
        )
    )

    return VaakDataset(
        manifest=manifest,
        split=split,
        audio_pipeline=pipeline,
        project_root=PROJECT_ROOT,
        random_crop=random_crop,
    )


def test_dataset_length() -> None:
    dataset = build_dataset("train")

    assert len(dataset) == 2


def test_dataset_returns_training_sample() -> None:
    dataset = build_dataset("train")

    sample = dataset[0]

    assert isinstance(sample["audio"], torch.Tensor)
    assert sample["audio"].shape == (64_000,)
    assert sample["audio"].dtype == torch.float32
    assert sample["label"] in {0, 1}


def test_dataset_preserves_metadata() -> None:
    dataset = build_dataset("train")

    sample = dataset[0]

    assert sample["sample_id"] == "sample_001"
    assert sample["speaker_id"] == "spk_001"
    assert sample["dataset"] == "demo"
    assert sample["attack_id"] is None


def test_dataset_preserves_spoof_attack_id() -> None:
    dataset = build_dataset("train")

    sample = dataset[1]

    assert sample["label"] == 1
    assert sample["attack_id"] == "A01"


def test_random_crop_returns_fixed_size() -> None:
    dataset = build_dataset(
        "train",
        random_crop=True,
    )

    sample = dataset[0]

    assert sample["audio"].shape == (64_000,)


def test_dataset_works_with_dataloader() -> None:
    dataset = build_dataset("train")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=vaak_collate_fn,
    )

    batch = next(iter(loader))

    assert batch["audio"].shape == (2, 64_000)
    assert batch["labels"].shape == (2,)

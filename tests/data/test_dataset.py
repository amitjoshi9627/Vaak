from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from vaak.data.dataset import VaakSample, vaak_collate_fn
from vaak.data.evaluation_dataset import EvaluationDataset
from vaak.data.training_dataset import TrainingDataset


@pytest.fixture
def dummy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "audio_path": ["dummy1.wav", "dummy2.wav", "dummy3.wav"],
            "label": [0, 1, 1],
            "sample_id": ["s1", "s2", "s3"],
            "speaker_id": ["spk1", "spk1", "spk2"],
            "dataset": ["asv2019", "asv2019", "asv2019"],
            "attack_id": [None, "A01", "A02"],
        }
    )


@pytest.fixture
def mock_audio_pipeline() -> MagicMock:
    pipeline = MagicMock()
    # Simulate processing an audio file into 3 overlapping 64000-sample chunks
    pipeline.process.return_value = torch.randn(3, 64000)
    return pipeline


def test_training_dataset_returns_single_chunk(
    dummy_manifest: pd.DataFrame, mock_audio_pipeline: MagicMock, tmp_path: Path
) -> None:
    dataset = TrainingDataset(
        manifest=dummy_manifest,
        split="train",
        audio_pipeline=mock_audio_pipeline,
        project_root=tmp_path,
    )

    sample = dataset[0]

    # Training should randomly pick 1 chunk out of the 3
    assert sample["audio"].shape == (64000,)
    assert sample["label"] == 0
    assert sample["attack_id"] is None


def test_evaluation_dataset_returns_all_chunks(
    dummy_manifest: pd.DataFrame, mock_audio_pipeline: MagicMock, tmp_path: Path
) -> None:
    dataset = EvaluationDataset(
        manifest=dummy_manifest,
        split="test",
        audio_pipeline=mock_audio_pipeline,
        project_root=tmp_path,
    )

    sample = dataset[0]

    # Evaluation must return the full sequence of chunks for utterance aggregation
    assert sample["audio"].shape == (3, 64000)
    assert sample["label"] == 1
    assert sample["attack_id"] == "A02"


def test_vaak_collate_fn() -> None:
    # Explicitly type the list so mypy knows these comply with the TypedDict
    samples: list[VaakSample] = [
        {
            "audio": torch.randn(64000),
            "label": 0,
            "sample_id": "s1",
            "speaker_id": "spk1",
            "dataset": "asv",
            "attack_id": None,
        },
        {
            "audio": torch.randn(64000),
            "label": 1,
            "sample_id": "s2",
            "speaker_id": "spk2",
            "dataset": "asv",
            "attack_id": "A01",
        },
    ]

    batch = vaak_collate_fn(samples)

    # Validate plural keys and tensor stacking
    assert batch["audio"].shape == (2, 64000)
    assert torch.equal(batch["labels"], torch.tensor([0, 1]))
    assert batch["attack_ids"] == [None, "A01"]
    assert batch["sample_ids"] == ["s1", "s2"]

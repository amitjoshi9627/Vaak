from pathlib import Path

import pandas as pd
import pytest

from vaak.data.manifest import load_manifest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "manifests" / "test_manifest.csv"
)


def test_load_valid_manifest() -> None:
    """A valid manifest should load successfully."""

    manifest = load_manifest(FIXTURE_PATH)

    assert len(manifest) == 4

    assert list(manifest.columns) == [
        "sample_id",
        "audio_path",
        "label",
        "speaker_id",
        "dataset",
        "split",
        "attack_id",
    ]

    assert list(manifest["label"]) == [0, 1, 0, 1]


def test_manifest_contains_expected_sample_ids() -> None:
    """Sample IDs should be preserved exactly."""

    manifest = load_manifest(FIXTURE_PATH)

    assert list(manifest["sample_id"]) == [
        "sample_001",
        "sample_002",
        "sample_003",
        "sample_004",
    ]


def test_manifest_preserves_attack_id() -> None:
    """Attack IDs should be preserved, including missing bona-fide IDs."""

    manifest = load_manifest(FIXTURE_PATH)

    assert manifest.iloc[0]["attack_id"] != manifest.iloc[1]["attack_id"]

    assert pd.isna(manifest.iloc[0]["attack_id"])
    assert manifest.iloc[1]["attack_id"] == "A01"


def test_manifest_requires_all_columns(tmp_path: Path) -> None:
    """Missing required columns should be rejected."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001"],
            "audio_path": ["a.wav"],
            "label": [0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_manifest(path)


def test_manifest_rejects_invalid_label(tmp_path: Path) -> None:
    """Labels other than 0 and 1 should be rejected."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001"],
            "audio_path": ["a.wav"],
            "label": [2],
            "speaker_id": ["spk1"],
            "dataset": ["demo"],
            "split": ["train"],
            "attack_id": ["A01"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match=r"Labels must be 0 .* or 1",
    ):
        load_manifest(path)


def test_manifest_rejects_invalid_split(tmp_path: Path) -> None:
    """Unknown split values should be rejected."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001"],
            "audio_path": ["a.wav"],
            "label": [0],
            "speaker_id": ["spk1"],
            "dataset": ["demo"],
            "split": ["unknown"],
            "attack_id": [None],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Split must",
    ):
        load_manifest(path)


def test_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    """Sample IDs must uniquely identify records."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001", "sample_001"],
            "audio_path": ["a.wav", "b.wav"],
            "label": [0, 1],
            "speaker_id": ["spk1", "spk2"],
            "dataset": ["demo", "demo"],
            "split": ["train", "test"],
            "attack_id": [None, "A01"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="sample_id values must be unique",
    ):
        load_manifest(path)


def test_manifest_rejects_missing_speaker_id(tmp_path: Path) -> None:
    """Speaker IDs are required for leakage-safe evaluation."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001"],
            "audio_path": ["a.wav"],
            "label": [0],
            "speaker_id": [None],
            "dataset": ["demo"],
            "split": ["train"],
            "attack_id": [None],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="speaker_id cannot contain missing values",
    ):
        load_manifest(path)


def test_manifest_rejects_missing_audio_path(tmp_path: Path) -> None:
    """Audio paths are required."""

    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "sample_id": ["sample_001"],
            "audio_path": [None],
            "label": [0],
            "speaker_id": ["spk1"],
            "dataset": ["demo"],
            "split": ["train"],
            "attack_id": [None],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="audio_path cannot contain missing values",
    ):
        load_manifest(path)

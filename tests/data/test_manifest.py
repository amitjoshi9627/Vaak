from pathlib import Path

import pandas as pd
import pytest

from vaak.data.manifest import load_manifest


def test_load_valid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "audio_path": ["a.wav", "b.wav"],
            "label": [0, 1],
            "speaker_id": ["spk1", "spk2"],
            "dataset": ["demo", "demo"],
            "split": ["train", "test"],
        }
    ).to_csv(path, index=False)

    manifest = load_manifest(path)

    assert len(manifest) == 2
    assert list(manifest["label"]) == [0, 1]


def test_manifest_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "label": [0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(path)


def test_manifest_rejects_invalid_label(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "label": [2],
            "speaker_id": ["spk1"],
            "dataset": ["demo"],
            "split": ["train"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Labels must"):
        load_manifest(path)


def test_manifest_rejects_invalid_split(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"

    pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "label": [0],
            "speaker_id": ["spk1"],
            "dataset": ["demo"],
            "split": ["unknown"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Split must"):
        load_manifest(path)

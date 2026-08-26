import pandas as pd
import pytest

from vaak.data.splits import split_by_speaker


def make_manifest() -> pd.DataFrame:
    rows = []

    for speaker_id in range(20):
        for sample_id in range(5):
            rows.append(
                {
                    "audio_path": f"{speaker_id}_{sample_id}.wav",
                    "label": sample_id % 2,
                    "speaker_id": f"speaker_{speaker_id}",
                    "dataset": "demo",
                    "split": "",
                }
            )

    return pd.DataFrame(rows)


def test_split_is_speaker_disjoint() -> None:
    manifest = make_manifest()

    result = split_by_speaker(
        manifest,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )

    train_speakers = set(result.loc[result["split"] == "train", "speaker_id"])

    val_speakers = set(result.loc[result["split"] == "val", "speaker_id"])

    test_speakers = set(result.loc[result["split"] == "test", "speaker_id"])

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)


def test_all_rows_are_assigned() -> None:
    manifest = make_manifest()

    result = split_by_speaker(manifest)

    assert len(result) == len(manifest)
    assert set(result["split"]) == {"train", "val", "test"}


def test_missing_speaker_id_is_rejected() -> None:
    manifest = make_manifest()
    manifest.loc[0, "speaker_id"] = None

    with pytest.raises(ValueError, match="speaker_id"):
        split_by_speaker(manifest)

from pathlib import Path

import pandas as pd

from vaak.data.adapters.asvspoof2019 import (
    ASVspoof2019LAAdapter,
)


def create_fake_asvspoof_structure(
    root: Path,
) -> None:
    """Create a tiny ASVspoof-like directory tree."""

    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True)

    for directory in (
        "ASVspoof2019_LA_train",
        "ASVspoof2019_LA_dev",
        "ASVspoof2019_LA_eval",
    ):
        (root / directory / "flac").mkdir(parents=True)

    train_protocol = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"

    train_protocol.write_text(
        "\n".join(
            [
                "LA_0001 LA_T_0000001 - - bonafide",
                "LA_0002 LA_T_0000002 - A01 spoof",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dev_protocol = protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt"

    dev_protocol.write_text(
        "LA_0003 LA_D_0000001 - A02 spoof\n",
        encoding="utf-8",
    )

    eval_protocol = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    eval_protocol.write_text(
        "LA_0004 LA_E_0000001 - - bonafide\n",
        encoding="utf-8",
    )


def test_build_manifest(tmp_path: Path) -> None:
    """Adapter should convert protocols to the canonical schema."""

    create_fake_asvspoof_structure(tmp_path)

    adapter = ASVspoof2019LAAdapter(tmp_path)

    manifest = adapter.build_manifest()

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


def test_labels_are_mapped_correctly(
    tmp_path: Path,
) -> None:
    """Bonafide should map to 0 and spoof to 1."""

    create_fake_asvspoof_structure(tmp_path)

    manifest = ASVspoof2019LAAdapter(tmp_path).build_manifest()

    train = manifest[manifest["split"] == "train"]

    bonafide = train[train["sample_id"] == "LA_T_0000001"].iloc[0]

    spoof = train[train["sample_id"] == "LA_T_0000002"].iloc[0]

    assert bonafide["label"] == 0
    assert pd.isna(bonafide["attack_id"])

    assert spoof["label"] == 1
    assert spoof["attack_id"] == "A01"


def test_splits_are_preserved(
    tmp_path: Path,
) -> None:
    """Official protocol partitions should map to train/val/test."""

    create_fake_asvspoof_structure(tmp_path)

    manifest = ASVspoof2019LAAdapter(tmp_path).build_manifest()

    assert set(manifest["split"]) == {
        "train",
        "val",
        "test",
    }


def test_paths_are_relative_to_repository(
    tmp_path: Path,
) -> None:
    """Manifest paths should not contain absolute local paths."""

    create_fake_asvspoof_structure(tmp_path)

    manifest = ASVspoof2019LAAdapter(tmp_path).build_manifest()

    for path in manifest["audio_path"]:
        assert not Path(path).is_absolute()


def test_dataset_name_is_correct(
    tmp_path: Path,
) -> None:
    create_fake_asvspoof_structure(tmp_path)

    manifest = ASVspoof2019LAAdapter(tmp_path).build_manifest()

    assert set(manifest["dataset"]) == {"asvspoof2019_la"}

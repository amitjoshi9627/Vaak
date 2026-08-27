from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "sample_id",
    "audio_path",
    "label",
    "speaker_id",
    "dataset",
    "split",
    "attack_id",
}

VALID_LABELS = {0, 1}
VALID_SPLITS = {"train", "val", "test"}


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate a Vaak manifest."""

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    manifest = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing_columns)}"
        )

    if manifest.empty:
        raise ValueError("Manifest is empty")

    if manifest["sample_id"].isna().any():
        raise ValueError("sample_id cannot contain missing values")

    if manifest["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")

    if manifest["audio_path"].isna().any():
        raise ValueError("audio_path cannot contain missing values")

    if manifest["speaker_id"].isna().any():
        raise ValueError("speaker_id cannot contain missing values")

    if manifest["dataset"].isna().any():
        raise ValueError("dataset cannot contain missing values")

    if manifest["label"].isna().any():
        raise ValueError("label cannot contain missing values")

    if not manifest["label"].isin(VALID_LABELS).all():
        raise ValueError("Labels must be 0 (bonafide) or 1 (spoof)")

    if manifest["split"].isna().any():
        raise ValueError("split cannot contain missing values")

    if not manifest["split"].isin(VALID_SPLITS).all():
        raise ValueError(f"Split must be one of: {sorted(VALID_SPLITS)}")

    return manifest

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AudioRecord:
    """Metadata describing one audio sample."""

    audio_path: Path
    label: int
    speaker_id: str
    dataset: str
    split: str


REQUIRED_COLUMNS = {
    "audio_path",
    "label",
    "speaker_id",
    "dataset",
    "split",
}


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate an audio manifest."""

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

    if not manifest["label"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 (real) or 1 (fake)")

    valid_splits = {"train", "val", "test"}

    if not manifest["split"].isin(valid_splits).all():
        raise ValueError(f"Split must be one of: {sorted(valid_splits)}")

    return manifest

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_speaker(
    manifest: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create speaker-disjoint train/validation/test splits."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    if "speaker_id" not in manifest.columns:
        raise ValueError("Manifest must contain speaker_id")

    if manifest["speaker_id"].isna().any():
        raise ValueError("speaker_id cannot contain missing values")

    working = manifest.copy()
    working["split"] = ""

    groups = working["speaker_id"]

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(first_split.split(working, groups=groups))

    working.loc[working.index[test_idx], "split"] = "test"

    train_val = working.iloc[train_val_idx]

    relative_val_size = val_size / (1.0 - test_size)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        second_split.split(
            train_val,
            groups=train_val["speaker_id"],
        )
    )

    train_indices = train_val.index[train_idx]
    val_indices = train_val.index[val_idx]

    working.loc[train_indices, "split"] = "train"
    working.loc[val_indices, "split"] = "val"

    return working

from pathlib import Path

import pandas as pd


class ASVspoof2019LAAdapter:
    """Convert ASVspoof 2019 Logical Access CM protocols
    into the canonical Vaak manifest format.
    """

    DATASET_NAME = "asvspoof2019_la"

    def __init__(self, root: Path) -> None:
        """
        Args:
            root:
                Path to data/raw/LA.
        """
        self.root = root

        self.protocol_dir = root / "ASVspoof2019_LA_cm_protocols"

        self.directories = {
            "train": "ASVspoof2019_LA_train",
            "val": "ASVspoof2019_LA_dev",
            "test": "ASVspoof2019_LA_eval",
        }

        self.protocols = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "val": "ASVspoof2019.LA.cm.dev.trl.txt",
            "test": "ASVspoof2019.LA.cm.eval.trl.txt",
        }

    def build_manifest(self) -> pd.DataFrame:
        """Build the complete train/val/test manifest."""

        frames = [self._parse_split(split) for split in ("train", "val", "test")]

        manifest = pd.concat(
            frames,
            ignore_index=True,
        )

        self._validate_manifest_integrity(manifest)

        return manifest

    def _parse_split(self, split: str) -> pd.DataFrame:
        protocol_name = self.protocols[split]

        protocol_path = self.protocol_dir / protocol_name

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        audio_dir = self.root / self.directories[split] / "flac"

        rows: list[dict[str, object]] = []

        with protocol_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) != 5:
                    raise ValueError(
                        f"Invalid protocol row in "
                        f"{protocol_name}:{line_number}. "
                        f"Expected 5 fields, got {len(fields)}: "
                        f"{line}"
                    )

                speaker_id = fields[0]
                sample_id = fields[1]
                unused_field = fields[2]
                system_id = fields[3]
                key = fields[4]

                if unused_field != "-":
                    raise ValueError(
                        f"Unexpected fourth field in "
                        f"{protocol_name}:{line_number}: "
                        f"{unused_field!r}"
                    )

                if key not in {"bonafide", "spoof"}:
                    raise ValueError(
                        f"Unexpected key in {protocol_name}:{line_number}: {key!r}"
                    )

                if key == "bonafide":
                    label = 0
                    attack_id = None

                    if system_id != "-":
                        raise ValueError(
                            f"Bonafide sample {sample_id} has "
                            f"unexpected system ID {system_id!r}"
                        )

                else:
                    label = 1
                    attack_id = system_id

                    if not system_id.startswith("A"):
                        raise ValueError(
                            f"Spoof sample {sample_id} has "
                            f"invalid attack ID {system_id!r}"
                        )

                audio_file = audio_dir / f"{sample_id}.flac"

                # Store paths relative to the repository root.
                relative_audio_path = audio_file.relative_to(self.root.parents[2])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": str(relative_audio_path),
                        "label": label,
                        "speaker_id": speaker_id,
                        "dataset": self.DATASET_NAME,
                        "split": split,
                        "attack_id": attack_id,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def _validate_manifest_integrity(
        manifest: pd.DataFrame,
    ) -> None:
        """Validate high-level adapter output."""

        required_columns = {
            "sample_id",
            "audio_path",
            "label",
            "speaker_id",
            "dataset",
            "split",
            "attack_id",
        }

        if not required_columns.issubset(manifest.columns):
            missing = required_columns - set(manifest.columns)

            raise ValueError(f"Adapter output missing columns: {sorted(missing)}")

        if manifest["sample_id"].duplicated().any():
            raise ValueError("ASVspoof manifest contains duplicate sample IDs")

        spoof_mask = manifest["label"] == 1

        if (
            manifest.loc[
                spoof_mask,
                "attack_id",
            ]
            .isna()
            .any()
        ):
            raise ValueError("Spoof samples must have an attack_id")

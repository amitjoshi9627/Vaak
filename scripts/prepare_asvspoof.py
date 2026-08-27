from pathlib import Path

from vaak.data.adapters.asvspoof2019 import ASVspoof2019LAAdapter
from vaak.data.manifest import load_manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_root = project_root / "data" / "raw" / "LA"

    output_path = project_root / "data" / "manifests" / "asvspoof2019_la.csv"

    adapter = ASVspoof2019LAAdapter(raw_root)

    print("Building ASVspoof 2019 LA manifest...")

    manifest = adapter.build_manifest()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_path,
        index=False,
    )

    print(f"\nManifest written to: {output_path}")

    validated = load_manifest(output_path)

    print("\nDataset summary")
    print("--" * 50)

    print(
        validated.groupby(
            ["split", "label"],
        ).size()
    )

    print("\nSpeakers:")
    print(validated.groupby("split")["speaker_id"].nunique())

    print("\nAttacks:")
    print(
        validated.loc[
            validated["label"] == 1,
            "attack_id",
        ]
        .value_counts()
        .sort_index()
    )

    print("\nTotal samples:")
    print(len(validated))


if __name__ == "__main__":
    main()

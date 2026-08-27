import argparse
from pathlib import Path

from vaak.data.adapters.asvspoof2019 import ASVspoof2019Adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    adapter = ASVspoof2019Adapter(args.root)

    manifest = adapter.build_manifest()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        args.output,
        index=False,
    )


if __name__ == "__main__":
    main()

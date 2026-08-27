from pathlib import Path

import pandas as pd


class ASVspoof2019Adapter:
    """Convert ASVspoof 2019 metadata into Vaak's manifest format."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def build_manifest(self) -> pd.DataFrame:
        """Build the canonical Vaak manifest."""

        raise NotImplementedError

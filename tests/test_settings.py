from pathlib import Path

import pytest

from vaak.config.settings import load_config
from vaak.core.exceptions import ConfigurationError


def test_load_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
project_name: vaak
version: "0.1.0"
environment: test
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.project_name == "vaak"
    assert config.version == "0.1.0"
    assert config.environment == "test"


def test_missing_config_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")

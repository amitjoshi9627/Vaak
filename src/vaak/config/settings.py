from dataclasses import dataclass
from pathlib import Path

import yaml

from vaak.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class VaakConfig:
    """Root Vaak configuration."""

    project_name: str
    version: str
    environment: str


def load_config(path: Path) -> VaakConfig:
    """Load Vaak configuration from a YAML file."""

    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping.")

    try:
        project_name = str(raw["project_name"])
        version = str(raw["version"])
        environment = str(raw["environment"])
    except KeyError as exc:
        raise ConfigurationError(f"Missing configuration field: {exc.args[0]}") from exc

    return VaakConfig(
        project_name=project_name,
        version=version,
        environment=environment,
    )

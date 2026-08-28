from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vaak.core.exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pretrained: str


class DataConfig(BaseModel):
    """Dataset and audio configuration."""

    model_config = ConfigDict(extra="forbid")

    manifest: Path
    sample_rate: int = Field(default=16_000, gt=0)
    chunk_duration_seconds: float = Field(default=4.0, gt=0)
    hop_duration_seconds: float = Field(default=2.0, gt=0)
    max_samples: int | None = None


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    epochs: int = Field(default=5, gt=0)


class VaakConfig(BaseModel):
    """Root configuration for a Vaak experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


def load_config(path: Path) -> VaakConfig:
    """Load and validate a Vaak YAML configuration."""

    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        return VaakConfig.model_validate(raw_config)

    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration: {path}") from exc

    except ValidationError as exc:
        raise ConfigurationError(f"Invalid Vaak configuration: {path}\n{exc}") from exc

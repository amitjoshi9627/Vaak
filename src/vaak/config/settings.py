from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vaak.core.exceptions import ConfigurationError


class LossStrategy(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"


class LayerStrategy(StrEnum):
    LAST = "last"
    WEIGHTED_SUM = "weighted_sum"


class PoolingStrategy(StrEnum):
    ASP = "asp"
    MEAN = "mean"


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pretrained_model_name: str
    layer_strategy: LayerStrategy = LayerStrategy.LAST
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    projection_dim: int | None = Field(
        default=None, gt=0, description="Optional bottleneck dimension"
    )


class DataConfig(BaseModel):
    """Dataset and audio configuration."""

    model_config = ConfigDict(extra="forbid")

    manifest: Path
    sample_rate: int = Field(default=16_000, gt=0)
    chunk_duration_seconds: float = Field(default=4.0, gt=0)
    hop_duration_seconds: float = Field(default=2.0, gt=0)
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, description="Random seed for reproducibility")
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    epochs: int = Field(default=5, gt=0)
    weight_decay: float = Field(default=1e-2, gt=0)
    loss_strategy: LossStrategy = LossStrategy.CROSS_ENTROPY


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

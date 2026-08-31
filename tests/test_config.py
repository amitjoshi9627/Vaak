from pathlib import Path

import pytest

from vaak.config.settings import load_config
from vaak.core.exceptions import ConfigurationError


def test_config_loads(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
            experiment_name: test_experiment

            model:
              name: model_name
              pretrained_model_name: models/model_name

            data:
              manifest: data/train.csv
              sample_rate: 16000
              chunk_duration_seconds: 4.0

            training:
              batch_size: 8
              learning_rate: 0.0001
              epochs: 5
            """,
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.experiment_name == "test_experiment"
    assert config.model.name == "model_name"
    assert config.data.sample_rate == 16_000
    assert config.training.batch_size == 8


def test_invalid_training_value_fails(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
            experiment_name: test_experiment

            model:
              name: model_name
              pretrained_model_name: models/model_name

            data:
              manifest: data/train.csv

            training:
              batch_size: -1
              learning_rate: 0.0001
              epochs: 5
            """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_config(config_file)


def test_unknown_field_fails(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
experiment_name: test_experiment

model:
  name: wavlm_base
  pretrained: microsoft/wavlm-base
  typo_field: true

data:
  manifest: data/train.csv

training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 5
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_config(config_file)


def test_missing_config_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")

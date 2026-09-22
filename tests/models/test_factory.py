from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaak.config.settings import (
    DataConfig,
    LayerStrategy,
    ModelConfig,
    PoolingStrategy,
    TrainingConfig,
    VaakConfig,
)
from vaak.models.backends.aggregation import WeightedLayerAggregation
from vaak.models.backends.pooling import AttentiveStatisticsPooling, MeanPooling
from vaak.models.factory import build_model_from_config


@pytest.fixture
def base_config() -> VaakConfig:
    return VaakConfig(
        experiment_name="test_experiment",
        model=ModelConfig(
            name="test_model",
            pretrained_model_name="dummy_pretrained",
            layer_strategy=LayerStrategy.LAST,
            pooling_strategy=PoolingStrategy.MEAN,
        ),
        data=DataConfig(manifest=Path("dummy.csv")),
        training=TrainingConfig(),
    )


@patch("vaak.models.factory.WavLMEncoder")
def test_factory_builds_baseline(
    mock_encoder_class: MagicMock, base_config: VaakConfig
) -> None:
    # Setup mock encoder dimensions
    mock_encoder = MagicMock()
    mock_encoder.output_dim = 768
    mock_encoder.num_hidden_states = 13
    mock_encoder_class.return_value = mock_encoder

    # Strategy: LAST + MEAN
    model = build_model_from_config(base_config)

    assert model.aggregator is None
    assert isinstance(model.pooler, MeanPooling)
    mock_encoder_class.assert_called_once_with(
        pretrained_name="dummy_pretrained",
        extract_all_layers=False,
        lora_r=8,
        lora_alpha=16,
    )


@patch("vaak.models.factory.WavLMEncoder")
def test_factory_builds_complex_architecture(
    mock_encoder_class: MagicMock, base_config: VaakConfig
) -> None:
    # Setup mock encoder dimensions
    mock_encoder = MagicMock()
    mock_encoder.output_dim = 768
    mock_encoder.num_hidden_states = 13
    mock_encoder_class.return_value = mock_encoder

    # Update config to complex setup
    base_config.model.layer_strategy = LayerStrategy.WEIGHTED_SUM
    base_config.model.pooling_strategy = PoolingStrategy.ASP

    model = build_model_from_config(base_config)

    assert isinstance(model.aggregator, WeightedLayerAggregation)
    assert isinstance(model.pooler, AttentiveStatisticsPooling)
    mock_encoder_class.assert_called_once_with(
        pretrained_name="dummy_pretrained",
        extract_all_layers=True,
        lora_r=8,
        lora_alpha=16,
    )

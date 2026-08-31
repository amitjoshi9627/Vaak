from pathlib import Path

import torch

from vaak.config.settings import (
    DataConfig,
    LayerStrategy,
    ModelConfig,
    PoolingStrategy,
    TrainingConfig,
    VaakConfig,
)
from vaak.models.factory import build_model_from_config


def test_forward_pass_complex_architecture() -> None:
    # 1. Create a dummy config for the complex V0.4 architecture
    config = VaakConfig(
        experiment_name="test_integration",
        model=ModelConfig(
            name="test_model",
            pretrained_model_name="microsoft/wavlm-base",
            layer_strategy=LayerStrategy.WEIGHTED_SUM,
            pooling_strategy=PoolingStrategy.ASP,
        ),
        data=DataConfig(manifest=Path("dummy.csv")),
        training=TrainingConfig(),
    )

    # 2. Build the model using the factory
    model = build_model_from_config(config)

    # 3. Pass a dummy training batch [Batch, Time]
    dummy_batch = torch.randn(2, 64_000)
    logits = model(dummy_batch)

    # 4. Verify logits shape [Batch, 2]
    assert logits.shape == (2, 2)

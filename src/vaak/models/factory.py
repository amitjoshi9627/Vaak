import torch.nn as nn

from vaak.config.settings import LayerStrategy, PoolingStrategy, VaakConfig
from vaak.models.backends.aggregation import WeightedLayerAggregation
from vaak.models.backends.pooling import AttentiveStatisticsPooling, MeanPooling
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.heads.binary import BinaryLinearHead


def build_model_from_config(config: VaakConfig) -> VaakDetector:
    """Instantiate a modular detector pipeline based on configuration."""

    layer_strategy = config.model.layer_strategy
    pooling_type = config.model.pooling_strategy

    extract_all = layer_strategy == LayerStrategy.WEIGHTED_SUM

    encoder = WavLMEncoder(
        pretrained_name=config.model.pretrained_model_name,
        freeze=True,
        extract_all_layers=extract_all,
    )

    aggregator = (
        WeightedLayerAggregation(num_layers=encoder.num_hidden_states)
        if extract_all
        else None
    )

    if pooling_type == PoolingStrategy.ASP:
        backend: nn.Module = AttentiveStatisticsPooling(
            input_dim=encoder.output_dim, attention_dim=128
        )
        head_dim = encoder.output_dim * 2
    else:
        backend = MeanPooling()
        head_dim = encoder.output_dim

    head = BinaryLinearHead(input_dim=head_dim)

    return VaakDetector(
        encoder=encoder,
        aggregator=aggregator,
        backend=backend,
        head=head,
    )

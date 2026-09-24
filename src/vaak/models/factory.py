import torch.nn as nn

from vaak.config.settings import (
    HeadStrategy,
    LayerStrategy,
    PoolingStrategy,
    VaakConfig,
)
from vaak.models.backends.aggregation import WeightedLayerAggregation
from vaak.models.backends.pooling import AttentiveStatisticsPooling, MeanPooling
from vaak.models.backends.projection import FrameProjection
from vaak.models.backends.temporal import TemporalCNN
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.heads.binary import BinaryLinearHead
from vaak.models.heads.cosine import CosineHead


def build_model_from_config(config: VaakConfig) -> VaakDetector:
    """Instantiate a modular detector pipeline based on configuration."""

    layer_strategy = config.model.layer_strategy
    pooling_type = config.model.pooling_strategy
    proj_dim = config.model.projection_dim

    extract_all = layer_strategy == LayerStrategy.WEIGHTED_SUM

    encoder = WavLMEncoder(
        pretrained_name=config.model.pretrained_model_name,
        extract_all_layers=extract_all,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
    )

    aggregator = (
        WeightedLayerAggregation(num_layers=encoder.num_hidden_states)
        if extract_all
        else None
    )

    # Calculate dimension coming out of encoder
    feature_dim = encoder.output_dim
    projector = None

    if proj_dim is not None:
        projector = FrameProjection(input_dim=encoder.output_dim, output_dim=proj_dim)
        feature_dim = proj_dim

    # Temporal layer
    temporal = TemporalCNN(channels=feature_dim)

    # Instantiate pooler with the correct incoming dimension
    if pooling_type == PoolingStrategy.ASP:
        pooler: nn.Module = AttentiveStatisticsPooling(
            input_dim=feature_dim, attention_dim=128
        )
        head_dim = feature_dim * 2
    else:
        pooler = MeanPooling()
        head_dim = feature_dim

    head_strategy = config.model.head_strategy

    if head_strategy == HeadStrategy.COSINE:
        head: nn.Module = CosineHead(input_dim=head_dim, scale=30.0)
    else:
        head = BinaryLinearHead(input_dim=head_dim)

    return VaakDetector(
        encoder=encoder,
        aggregator=aggregator,
        projector=projector,
        temporal=temporal,
        pooler=pooler,
        head=head,
    )

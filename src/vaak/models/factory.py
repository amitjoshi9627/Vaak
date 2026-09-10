import torch.nn as nn

from vaak.config.settings import LayerStrategy, PoolingStrategy, VaakConfig
from vaak.models.backends.aggregation import WeightedLayerAggregation
from vaak.models.backends.pooling import AttentiveStatisticsPooling, MeanPooling
from vaak.models.backends.projection import FrameProjection
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.heads.binary import BinaryLinearHead


def build_model_from_config(config: VaakConfig) -> VaakDetector:
    """Instantiate a modular detector pipeline based on configuration."""

    layer_strategy = config.model.layer_strategy
    pooling_type = config.model.pooling_strategy
    proj_dim = config.model.projection_dim

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

    # Calculate dimension transitioning into the backend
    backend_input_dim = encoder.output_dim
    projector = None

    if proj_dim is not None:
        projector = FrameProjection(input_dim=encoder.output_dim, output_dim=proj_dim)
        backend_input_dim = proj_dim

    # Instantiate pooling with the correct incoming dimension
    if pooling_type == PoolingStrategy.ASP:
        backend: nn.Module = AttentiveStatisticsPooling(
            input_dim=backend_input_dim, attention_dim=128
        )
        head_dim = backend_input_dim * 2
    else:
        backend = MeanPooling()
        head_dim = backend_input_dim

    head = BinaryLinearHead(input_dim=head_dim)

    return VaakDetector(
        encoder=encoder,
        aggregator=aggregator,
        projector=projector,
        backend=backend,
        head=head,
    )

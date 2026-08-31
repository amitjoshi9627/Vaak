import torch

from vaak.models.backends.aggregation import WeightedLayerAggregation


def test_weighted_layer_aggregation() -> None:
    # Setup dimensions: [num_layers, Batch, Time, Features]
    num_layers = 13
    batch_size = 2
    time_steps = 50
    feature_dim = 768

    stacked_states = torch.randn(num_layers, batch_size, time_steps, feature_dim)
    aggregator = WeightedLayerAggregation(num_layers=num_layers)

    fused_states = aggregator(stacked_states)

    # 1. Check Output Shape
    assert fused_states.shape == (batch_size, time_steps, feature_dim)

    # 2. Check Gradients (Alpha weights must be learnable)
    assert aggregator.weights.requires_grad
    assert aggregator.weights.shape == (num_layers,)

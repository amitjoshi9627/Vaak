import torch

from vaak.models.backends.temporal import TemporalCNN


def test_temporal_cnn_forward_pass() -> None:
    batch_size = 4
    time_steps = 50
    channels = 768

    model = TemporalCNN(channels=channels)

    # Input tensor shape: [B, T, D]
    x = torch.randn(batch_size, time_steps, channels)
    out = model(x)

    # Output should have the same shape due to padding
    assert out.shape == (batch_size, time_steps, channels)


def test_temporal_cnn_different_hidden_channels() -> None:
    batch_size = 2
    time_steps = 30
    channels = 256
    hidden_channels = 512

    model = TemporalCNN(channels=channels, hidden_channels=hidden_channels)

    x = torch.randn(batch_size, time_steps, channels)
    out = model(x)

    assert out.shape == (batch_size, time_steps, channels)

import torch

from vaak.models.backends.projection import FrameProjection


def test_frame_projection_forward_pass() -> None:
    batch_size = 4
    time_steps = 100
    input_dim = 768
    output_dim = 256

    model = FrameProjection(input_dim=input_dim, output_dim=output_dim)

    # Input tensor shape: [B, T, input_dim]
    x = torch.randn(batch_size, time_steps, input_dim)
    out = model(x)

    # Output should project to output_dim
    assert out.shape == (batch_size, time_steps, output_dim)

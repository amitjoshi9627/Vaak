from unittest.mock import MagicMock, patch

import torch

from vaak.models.encoders.wavlm import WavLMEncoder


@patch("vaak.models.encoders.wavlm.WavLMModel.from_pretrained")
def test_wavlm_encoder_multi_layer(mock_from_pretrained: MagicMock) -> None:
    batch_size = 2
    time_steps = 50
    feature_dim = 768

    # Setup the Hugging Face mock to return a tuple of 13 hidden states
    mock_model_instance = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = tuple(
        torch.randn(batch_size, time_steps, feature_dim) for _ in range(13)
    )
    mock_model_instance.return_value = mock_outputs

    # Add dummy parameters to test the freezing logic
    mock_model_instance.parameters.return_value = [
        torch.nn.Parameter(torch.randn(10)) for _ in range(3)
    ]
    mock_from_pretrained.return_value = mock_model_instance

    encoder = WavLMEncoder(
        pretrained_name="dummy", freeze=True, extract_all_layers=True
    )
    dummy_audio = torch.randn(batch_size, 16000)

    stacked_states = encoder(dummy_audio)

    # 1. Check Output Shape
    assert stacked_states.shape == (13, batch_size, time_steps, feature_dim)

    # 2. Verify all WavLM parameters are explicitly frozen
    for param in encoder.model.parameters():
        assert not param.requires_grad

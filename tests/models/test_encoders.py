from unittest.mock import MagicMock, patch

import torch

from vaak.models.encoders.wavlm import WavLMEncoder


@patch("vaak.models.encoders.wavlm.WavLMModel.from_pretrained")
def test_wavlm_encoder_multi_layer(mock_from_pretrained: MagicMock) -> None:
    batch_size = 2
    time_steps = 50
    feature_dim = 768

    class DummyWavLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.randn(10))
            self.param2 = torch.nn.Parameter(torch.randn(10))
            self.param3 = torch.nn.Parameter(torch.randn(10))
            self.q_proj = torch.nn.Linear(10, 10)
            self.v_proj = torch.nn.Linear(10, 10)
            self.config = MagicMock()
            self.config.hidden_size = feature_dim
            self.config.num_hidden_layers = 12

        def forward(self, x: torch.Tensor) -> MagicMock:
            mock_outputs = MagicMock()
            mock_outputs.hidden_states = tuple(
                torch.randn(batch_size, time_steps, feature_dim) for _ in range(13)
            )
            return mock_outputs

    mock_model_instance = DummyWavLM()
    mock_from_pretrained.return_value = mock_model_instance

    encoder = WavLMEncoder(pretrained_name="dummy", extract_all_layers=True)
    dummy_audio = torch.randn(batch_size, 16000)

    stacked_states = encoder(dummy_audio)

    # 1. Check Output Shape
    assert stacked_states.shape == (13, batch_size, time_steps, feature_dim)

    # 2. Verify all WavLM base parameters are explicitly frozen, while LoRA parameters are trainable
    for name, param in encoder.model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

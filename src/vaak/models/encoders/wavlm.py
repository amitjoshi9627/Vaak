from typing import cast

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMEncoder(nn.Module):
    """WavLM foundation model with toggleable layer extraction."""

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-base",
        freeze: bool = True,
        extract_all_layers: bool = False,
    ) -> None:
        super().__init__()
        self.extract_all_layers = extract_all_layers

        self.model = WavLMModel.from_pretrained(
            pretrained_name,
            output_hidden_states=extract_all_layers,
            layerdrop=0.0,
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: Audio waveform tensor of shape [B, L].

        Returns:
            Stacked hidden states [num_layers, B, T, D] if extract_all_layers is True.
            Last hidden state [B, T, D] if extract_all_layers is False.
        """
        outputs = self.model(x)
        if self.extract_all_layers:
            return torch.stack(outputs.hidden_states, dim=0)
        return cast(torch.Tensor, outputs.last_hidden_state)

    @property
    def output_dim(self) -> int:
        """Feature dimension produced by WavLM."""
        return self.model.config.hidden_size

    @property
    def num_hidden_states(self) -> int:
        """Number of hidden states exposed by the encoder."""
        return self.model.config.num_hidden_layers + 1

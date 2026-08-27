from typing import cast

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMEncoder(nn.Module):
    """WavLM-based acoustic encoder."""

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-base",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(pretrained_name)

        if freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute acoustic representations.

        Args:
            x: Audio tensor of shape [batch_size, sequence_length].

        Returns:
            Hidden states of shape [batch_size, sequence_length, hidden_size].
        """
        outputs = self.wavlm(x)

        return cast(torch.Tensor, outputs.last_hidden_state)

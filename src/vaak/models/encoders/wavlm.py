from typing import cast

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import WavLMModel


class WavLMEncoder(nn.Module):
    """WavLM foundation model with LoRA adaptation."""

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-base",
        extract_all_layers: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.extract_all_layers = extract_all_layers

        # Load the frozen base model
        base_model = WavLMModel.from_pretrained(
            pretrained_name,
            output_hidden_states=extract_all_layers,
            layerdrop=0.0,
        )

        # Configure Low-Rank Adapters (LoRA)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )

        # Wrap model to inject trainable matrices and freeze the rest
        self.model = get_peft_model(base_model, config)

        self.model.print_trainable_parameters()

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

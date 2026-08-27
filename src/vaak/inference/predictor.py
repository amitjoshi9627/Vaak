from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from vaak.audio.pipeline import AudioPipeline
from vaak.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured output for a single audio file prediction."""

    is_spoof: bool
    spoof_probability: float
    chunk_probabilities: list[float]


class VaakPredictor:
    """High-level inference API for the Vaak web backend."""

    def __init__(
        self,
        model: nn.Module,
        audio_pipeline: AudioPipeline,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.audio_pipeline = audio_pipeline
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, audio_path: Path) -> PredictionResult:
        """Run end-to-end inference on an audio file."""
        logger.info(f"Running inference on: {audio_path.name}")

        chunks = self.audio_pipeline.process(audio_path).to(self.device)

        logits = cast(torch.Tensor, self.model(chunks))

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

        global_prob = sum(probs) / len(probs) if probs else 0.0

        is_spoof = global_prob >= self.threshold

        return PredictionResult(
            is_spoof=is_spoof,
            spoof_probability=global_prob,
            chunk_probabilities=probs,
        )

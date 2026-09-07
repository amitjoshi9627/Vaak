from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from vaak.audio.pipeline import AudioPipeline
from vaak.config.constants import DECISION_THRESHOLD
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
        threshold: float = DECISION_THRESHOLD,
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

        probs = []
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logits = cast(torch.Tensor, self.model(batch))
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            probs.extend(batch_probs)

        global_prob = sum(probs) / len(probs) if probs else 0.0

        is_spoof = global_prob >= self.threshold

        return PredictionResult(
            is_spoof=is_spoof,
            spoof_probability=global_prob,
            chunk_probabilities=probs,
        )

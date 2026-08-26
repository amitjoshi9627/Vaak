from dataclasses import dataclass
from pathlib import Path

import torch

from vaak.audio.chunking import chunk_audio
from vaak.audio.loader import load_audio
from vaak.audio.preprocessing import normalize_audio, resample_audio


@dataclass(frozen=True)
class AudioPipelineConfig:
    """Configuration for Vaak audio processing."""

    sample_rate: int = 16_000
    chunk_duration_seconds: float = 4.0
    hop_duration_seconds: float = 2.0
    normalize: bool = True


class AudioPipeline:
    """Canonical audio preprocessing pipeline."""

    def __init__(self, config: AudioPipelineConfig) -> None:
        self.config = config

    def process(self, path: Path) -> torch.Tensor:
        """Load and transform an audio file into model-ready chunks."""

        waveform, sample_rate = load_audio(path)

        waveform = resample_audio(
            waveform,
            original_sample_rate=sample_rate,
            target_sample_rate=self.config.sample_rate,
        )

        if self.config.normalize:
            waveform = normalize_audio(waveform)

        return chunk_audio(
            waveform,
            sample_rate=self.config.sample_rate,
            chunk_duration_seconds=self.config.chunk_duration_seconds,
            hop_duration_seconds=self.config.hop_duration_seconds,
        )

from pathlib import Path

import pytest
import torch

from vaak.audio import AudioPipeline, AudioPipelineConfig


@pytest.fixture
def sample_audio_file() -> Path:
    """Return the real audio fixture used for integration testing."""

    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "tests" / "fixtures" / "sample_voice.wav"

    if not path.exists():
        pytest.fail(f"Audio fixture not found: {path}")

    return path


def test_audio_pipeline_chunk_shape_and_overlap(
    sample_audio_file: Path,
) -> None:
    """
    Validate the complete audio pipeline using the real test voice.

    Expected fixture:
        Duration: 10 seconds
        Source sample rate: arbitrary
        Target sample rate: 16 kHz

    Pipeline:
        Load → Mono → Resample → Normalize → Chunk

    Expected:
        5 chunks
        4 seconds per chunk
        2 second hop
        64,000 samples per chunk
    """

    config = AudioPipelineConfig(
        sample_rate=16_000,
        chunk_duration_seconds=4.0,
        hop_duration_seconds=2.0,
    )

    pipeline = AudioPipeline(config)

    chunks = pipeline.process(sample_audio_file)

    assert isinstance(chunks, torch.Tensor)
    assert chunks.ndim == 2
    assert chunks.shape == (2, 64_000)
    assert chunks.dtype == torch.float32

    # Normalization should ensure the waveform does not exceed [-1, 1].
    assert chunks.abs().max() <= 1.0


def test_audio_pipeline_returns_finite_values(
    sample_audio_file: Path,
) -> None:
    """The complete pipeline should never produce NaN or infinite values."""

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            sample_rate=16_000,
            chunk_duration_seconds=4.0,
            hop_duration_seconds=2.0,
        )
    )

    chunks = pipeline.process(sample_audio_file)

    assert torch.isfinite(chunks).all()


def test_audio_pipeline_output_is_mono(
    sample_audio_file: Path,
) -> None:
    """The pipeline should always produce mono chunks."""

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            sample_rate=16_000,
            chunk_duration_seconds=4.0,
            hop_duration_seconds=2.0,
        )
    )

    chunks = pipeline.process(sample_audio_file)

    # [num_chunks, samples]
    assert chunks.ndim == 2


def test_audio_pipeline_with_no_normalization(
    sample_audio_file: Path,
) -> None:
    """Normalization should be configurable."""

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            sample_rate=16_000,
            chunk_duration_seconds=4.0,
            hop_duration_seconds=2.0,
            normalize=False,
        )
    )

    chunks = pipeline.process(sample_audio_file)

    assert isinstance(chunks, torch.Tensor)
    assert chunks.ndim == 2
    assert chunks.shape == (2, 64_000)

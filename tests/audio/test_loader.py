from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vaak.audio.loader import load_audio


def test_load_mono_audio(tmp_path: Path) -> None:
    """Load a mono WAV file and return a 1D float tensor."""

    path = tmp_path / "mono.wav"

    waveform = np.random.randn(16_000).astype(np.float32)

    sf.write(
        path,
        waveform,
        16_000,
    )

    loaded, sample_rate = load_audio(path)

    assert loaded.ndim == 1
    assert loaded.shape == (16_000,)
    assert loaded.dtype == torch.float32
    assert sample_rate == 16_000


def test_stereo_audio_is_converted_to_mono(tmp_path: Path) -> None:
    """Multi-channel audio should be converted to mono."""

    path = tmp_path / "stereo.wav"

    waveform = np.stack(
        [
            np.ones(16_000, dtype=np.float32),
            np.zeros(16_000, dtype=np.float32),
        ],
        axis=1,
    )

    sf.write(
        path,
        waveform,
        16_000,
    )

    loaded, sample_rate = load_audio(path)

    assert loaded.ndim == 1
    assert loaded.shape == (16_000,)
    assert loaded.dtype == torch.float32
    assert torch.allclose(
        loaded,
        torch.full((16_000,), 0.5),
        atol=1e-4,
    )
    assert sample_rate == 16_000


def test_load_audio_returns_float32(tmp_path: Path) -> None:
    """Audio returned by the loader should always be float32."""

    path = tmp_path / "audio.wav"

    waveform = np.random.randn(8_000).astype(np.float32)

    sf.write(
        path,
        waveform,
        8_000,
    )

    loaded, _ = load_audio(path)

    assert loaded.dtype == torch.float32

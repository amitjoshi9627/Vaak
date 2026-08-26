import pytest
import torch

from vaak.audio.preprocessing import normalize_audio, resample_audio


def test_normalize_audio() -> None:
    """Peak normalization should scale the maximum absolute value to 1."""

    waveform = torch.tensor([0.5, -1.0, 0.25])

    normalized = normalize_audio(waveform)

    assert torch.isclose(
        normalized.abs().max(),
        torch.tensor(1.0),
    )

    assert torch.allclose(
        normalized,
        torch.tensor([0.5, -1.0, 0.25]),
    )


def test_normalize_audio_scales_waveform() -> None:
    """Normalization should preserve the relative waveform shape."""

    waveform = torch.tensor([0.25, -0.5, 0.125])

    normalized = normalize_audio(waveform)

    expected = torch.tensor([0.5, -1.0, 0.25])

    assert torch.allclose(normalized, expected)


def test_silent_audio_is_unchanged() -> None:
    """A silent waveform should not produce NaNs during normalization."""

    waveform = torch.zeros(100)

    normalized = normalize_audio(waveform)

    assert torch.equal(normalized, waveform)
    assert not torch.isnan(normalized).any()


def test_resample_audio() -> None:
    """Audio should be resampled to the requested sample rate."""

    waveform = torch.randn(16_000)

    resampled = resample_audio(
        waveform,
        original_sample_rate=16_000,
        target_sample_rate=8_000,
    )

    assert resampled.ndim == 1
    assert resampled.shape == (8_000,)
    assert resampled.dtype == torch.float32


def test_resample_audio_same_sample_rate() -> None:
    """No resampling should occur when sample rates already match."""

    waveform = torch.randn(16_000)

    resampled = resample_audio(
        waveform,
        original_sample_rate=16_000,
        target_sample_rate=16_000,
    )

    assert torch.equal(resampled, waveform)


def test_resample_audio_8k_to_16k() -> None:
    """8 kHz audio should approximately double its sample count."""

    waveform = torch.randn(8_000)

    resampled = resample_audio(
        waveform,
        original_sample_rate=8_000,
        target_sample_rate=16_000,
    )

    assert resampled.shape == (16_000,)
    assert resampled.dtype == torch.float32


@pytest.mark.parametrize(
    ("original_rate", "target_rate"),
    [
        (0, 16_000),
        (-1, 16_000),
        (16_000, 0),
        (16_000, -1),
    ],
)
def test_resample_audio_rejects_invalid_sample_rates(
    original_rate: int,
    target_rate: int,
) -> None:
    """Invalid sample rates should fail explicitly."""

    waveform = torch.randn(1_000)

    with pytest.raises(ValueError):
        resample_audio(
            waveform,
            original_sample_rate=original_rate,
            target_sample_rate=target_rate,
        )


def test_resample_audio_rejects_non_1d_waveform() -> None:
    """Resampling should only accept a single waveform."""

    waveform = torch.randn(2, 16_000)

    with pytest.raises(ValueError):
        resample_audio(
            waveform,
            original_sample_rate=16_000,
            target_sample_rate=8_000,
        )

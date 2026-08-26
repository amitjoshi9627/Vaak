import torch
import torch.nn.functional as F


def resample_audio(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Resample a 1-D waveform to the target sample rate."""

    if original_sample_rate <= 0:
        raise ValueError("original_sample_rate must be positive.")

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive.")

    if waveform.ndim != 1:
        raise ValueError(f"Expected a 1-D waveform, got shape {tuple(waveform.shape)}.")

    if original_sample_rate == target_sample_rate:
        return waveform

    original_length = waveform.shape[0]

    target_length = round(original_length * target_sample_rate / original_sample_rate)

    # [samples] -> [1, 1, samples]
    audio = waveform.float().unsqueeze(0).unsqueeze(0)

    resampled = F.interpolate(
        audio,
        size=target_length,
        mode="linear",
        align_corners=False,
    )

    # [1, 1, samples] -> [samples]
    return resampled.squeeze(0).squeeze(0)


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize audio.

    The waveform is scaled so that its maximum absolute amplitude
    is 1.0. Silent audio is returned unchanged.
    """

    peak = waveform.abs().max()

    if peak == 0:
        return waveform

    return waveform / peak

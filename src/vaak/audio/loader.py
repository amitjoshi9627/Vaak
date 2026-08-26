from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """
    Load an audio file into a float32 PyTorch tensor.

    Returns:
        waveform: 1-D tensor of shape [num_samples].
        sample_rate: Original sample rate.
    """
    try:
        audio, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file: {path}") from exc

    waveform = torch.from_numpy(audio)

    # Convert multi-channel audio to mono.
    waveform = waveform.mean(dim=1) if waveform.shape[1] > 1 else waveform[:, 0]

    return waveform.contiguous(), sample_rate

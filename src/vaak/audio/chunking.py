import torch
import torch.nn.functional as F


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_seconds: float,
    hop_duration_seconds: float | None = None,
) -> torch.Tensor:
    """Split a mono waveform into fixed-length overlapping chunks."""

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected mono waveform [samples], got {tuple(waveform.shape)}"
        )

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    if chunk_duration_seconds <= 0:
        raise ValueError("chunk_duration_seconds must be > 0")

    if hop_duration_seconds is None:
        hop_duration_seconds = chunk_duration_seconds

    if hop_duration_seconds <= 0:
        raise ValueError("hop_duration_seconds must be > 0")

    chunk_samples = round(sample_rate * chunk_duration_seconds)
    hop_samples = round(sample_rate * hop_duration_seconds)

    if waveform.numel() <= chunk_samples:
        padded = F.pad(
            waveform,
            (0, chunk_samples - waveform.numel()),
        )
        return padded.unsqueeze(0)

    chunks: list[torch.Tensor] = []

    for start in range(0, waveform.numel(), hop_samples):
        end = start + chunk_samples

        chunk = waveform[start:end]

        if chunk.numel() < chunk_samples:
            chunk = F.pad(
                chunk,
                (0, chunk_samples - chunk.numel()),
            )

        chunks.append(chunk)

        if end >= waveform.numel():
            break

    return torch.stack(chunks)

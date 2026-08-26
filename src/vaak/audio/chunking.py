import torch


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_seconds: float,
    hop_duration_seconds: float | None = None,
) -> torch.Tensor:
    """Split audio into fixed-length overlapping chunks.

    Args:
        waveform: Mono waveform with shape [num_samples].
        sample_rate: Audio sample rate.
        chunk_duration_seconds: Duration of each chunk.
        hop_duration_seconds: Distance between chunk starts.
            Defaults to chunk duration (no overlap).

    Returns:
        Tensor with shape [num_chunks, chunk_samples].
    """

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected mono waveform [samples], got {tuple(waveform.shape)}"
        )

    if chunk_duration_seconds <= 0:
        raise ValueError("chunk_duration_seconds must be > 0")

    if hop_duration_seconds is None:
        hop_duration_seconds = chunk_duration_seconds

    if hop_duration_seconds <= 0:
        raise ValueError("hop_duration_seconds must be > 0")

    chunk_samples = round(sample_rate * chunk_duration_seconds)
    hop_samples = round(sample_rate * hop_duration_seconds)

    if waveform.numel() <= chunk_samples:
        padded = torch.nn.functional.pad(
            waveform,
            (0, chunk_samples - waveform.numel()),
        )
        return padded.unsqueeze(0)

    chunks = []

    for start in range(0, waveform.numel(), hop_samples):
        end = start + chunk_samples

        chunk = waveform[start:end]

        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, chunk_samples - chunk.numel()),
            )

        chunks.append(chunk)

        if end >= waveform.numel():
            break

    return torch.stack(chunks)

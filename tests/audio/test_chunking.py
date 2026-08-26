import torch

from vaak.audio.chunking import chunk_audio


def test_short_audio_is_padded() -> None:
    """Audio shorter than the chunk size should be zero-padded."""

    waveform = torch.ones(4_000)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
    )

    assert chunks.shape == (1, 8_000)

    assert torch.equal(
        chunks[0, :4_000],
        waveform,
    )

    assert torch.all(
        chunks[0, 4_000:] == 0,
    )


def test_exact_length_audio_produces_one_chunk() -> None:
    """Audio exactly equal to the chunk size should produce one chunk."""

    waveform = torch.randn(8_000)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
    )

    assert chunks.shape == (1, 8_000)


def test_audio_longer_than_chunk_produces_multiple_chunks() -> None:
    """Long audio should be split into multiple chunks."""

    waveform = torch.randn(16_000)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
    )

    assert chunks.shape == (2, 8_000)


def test_overlapping_chunks() -> None:
    """50% overlap should produce the expected number of chunks."""

    waveform = torch.randn(16_000)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
        hop_duration_seconds=0.5,
    )

    assert chunks.shape == (3, 8_000)


def test_overlap_preserves_audio_content() -> None:
    """Overlapping chunks should contain the same samples in the overlap."""

    waveform = torch.arange(16_000, dtype=torch.float32)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
        hop_duration_seconds=0.5,
    )

    # First chunk: samples [0, 8000)
    # Second chunk: samples [4000, 12000)
    assert torch.equal(
        chunks[0, 4_000:],
        chunks[1, :4_000],
    )


def test_last_chunk_is_padded() -> None:
    """The final incomplete chunk should be zero-padded."""

    waveform = torch.ones(12_000)

    chunks = chunk_audio(
        waveform,
        sample_rate=8_000,
        chunk_duration_seconds=1.0,
        hop_duration_seconds=1.0,
    )

    assert chunks.shape == (2, 8_000)

    # First chunk is complete.
    assert torch.all(chunks[0] == 1)

    # Second chunk contains 4,000 real samples + 4,000 padding.
    assert torch.all(chunks[1, :4_000] == 1)
    assert torch.all(chunks[1, 4_000:] == 0)


def test_chunk_audio_rejects_non_mono_input() -> None:
    """Chunking should only accept [samples] tensors."""

    waveform = torch.randn(2, 8_000)

    try:
        chunk_audio(
            waveform,
            sample_rate=8_000,
            chunk_duration_seconds=1.0,
        )
    except ValueError as exc:
        assert "Expected mono waveform" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_audio_rejects_invalid_chunk_duration() -> None:
    """Chunk duration must be positive."""

    waveform = torch.randn(8_000)

    try:
        chunk_audio(
            waveform,
            sample_rate=8_000,
            chunk_duration_seconds=0,
        )
    except ValueError as exc:
        assert "chunk_duration_seconds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_audio_rejects_invalid_hop_duration() -> None:
    """Hop duration must be positive."""

    waveform = torch.randn(8_000)

    try:
        chunk_audio(
            waveform,
            sample_rate=8_000,
            chunk_duration_seconds=1.0,
            hop_duration_seconds=0,
        )
    except ValueError as exc:
        assert "hop_duration_seconds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

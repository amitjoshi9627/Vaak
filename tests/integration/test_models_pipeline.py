import torch

from vaak.models.backends.pooling import MeanPooling
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.heads.binary import BinaryLinearHead


def test_forward_pass() -> None:
    encoder = WavLMEncoder(pretrained_name="microsoft/wavlm-base", freeze=True)
    backend = MeanPooling()
    head = BinaryLinearHead(input_dim=768)

    model = VaakDetector(encoder=encoder, backend=backend, head=head)

    dummy_batch = torch.randn(2, 64_000)
    logits = model(dummy_batch)

    assert logits.shape == (2, 2)

import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vaak.audio.pipeline import AudioPipeline, AudioPipelineConfig
from vaak.core.logging import get_logger
from vaak.inference.predictor import VaakPredictor
from vaak.models.backends.pooling import MeanPooling
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.heads.binary import BinaryLinearHead
from vaak.utils.tools import get_optimal_device

predictor: VaakPredictor | None = None

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the application lifecycle and model initialization."""
    global predictor

    device = get_optimal_device()

    encoder = WavLMEncoder(freeze=True)
    backend = MeanPooling()
    head = BinaryLinearHead(input_dim=768)
    model = VaakDetector(encoder=encoder, backend=backend, head=head)

    # Load the best model dynamically
    project_root = Path(__file__).resolve().parents[3]
    champion_path = project_root / "registry" / "champion_model.pt"

    if champion_path.exists():
        logger.info(f"Loading champion model from {champion_path}")
        checkpoint = torch.load(champion_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning(
            "No champion model found in registry! Initializing with untrained weights."
        )

    pipeline = AudioPipeline(config=AudioPipelineConfig())

    predictor = VaakPredictor(
        model=model,
        audio_pipeline=pipeline,
        device=device,
    )

    yield

    predictor = None


app = FastAPI(
    title="Vaak API",
    description="AI-powered speech forensics API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    """API response schema for audio predictions."""

    is_spoof: bool
    spoof_probability: float
    chunk_probabilities: list[float]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verify API status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_audio(
    file: Annotated[UploadFile, File()],
) -> PredictResponse:
    """Analyze an uploaded audio file for synthetic artifacts."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix
    if suffix.lower() not in {".wav", ".flac"}:
        raise HTTPException(
            status_code=400,
            detail="Only .wav and .flac files are supported",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        try:
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            result = predictor.predict(tmp_path)

            return PredictResponse(
                is_spoof=result.is_spoof,
                spoof_probability=result.spoof_probability,
                chunk_probabilities=result.chunk_probabilities,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

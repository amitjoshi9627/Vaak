import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vaak import __version__
from vaak.audio.pipeline import AudioPipeline, AudioPipelineConfig
from vaak.config.settings import load_config
from vaak.core.logging import get_logger
from vaak.inference.predictor import VaakPredictor
from vaak.models.backends.pooling import MeanPooling
from vaak.models.detector import VaakDetector
from vaak.models.encoders.wavlm import WavLMEncoder
from vaak.models.factory import build_model_from_config
from vaak.models.heads.binary import BinaryLinearHead
from vaak.utils.tools import get_optimal_device

predictor: VaakPredictor | None = None

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the application lifecycle and dynamic model initialization."""
    global predictor

    device = get_optimal_device()
    project_root = Path(__file__).resolve().parents[3]

    champion_path = project_root / "registry" / "champion_model.pt"
    champion_config_path = project_root / "registry" / "champion_config.yaml"

    try:
        if champion_path.exists() and champion_config_path.exists():
            logger.info(f"Loading champion config from {champion_config_path}")
            config = load_config(champion_config_path)

            model = build_model_from_config(config)

            logger.info(f"Loading champion weights from {champion_path}")
            checkpoint = torch.load(
                champion_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError("Champion model or config missing")
    except Exception as e:
        logger.warning(
            f"Failed to load champion model ({e})! Falling back to untrained baseline."
        )

        model = VaakDetector(
            encoder=WavLMEncoder(freeze=True),
            backend=MeanPooling(),
            head=BinaryLinearHead(input_dim=768),
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
    version=__version__,
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
    """Verify API status and version."""
    return {"status": "ok", "version": __version__}


@app.post("/predict", response_model=PredictResponse)
async def predict_audio(
    file: Annotated[UploadFile, File()],
) -> PredictResponse:
    """Analyze an uploaded audio file for synthetic artifacts."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix or ".tmp"

    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in,
        tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out,
    ):
        tmp_in_path = Path(tmp_in.name)
        tmp_out_path = Path(tmp_out.name)

        try:
            import shutil

            shutil.copyfileobj(file.file, tmp_in)
            tmp_in.flush()
            tmp_in.close()
            tmp_out.close()

            # Sanitize and truncate audio to max 120s
            import subprocess

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(tmp_in_path),
                        "-t",
                        "120",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(tmp_out_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                target_path = tmp_out_path
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode audio format. Please ensure it is a valid audio file.",
                ) from e

            result = predictor.predict(target_path)

            return PredictResponse(
                is_spoof=result.is_spoof,
                spoof_probability=result.spoof_probability,
                chunk_probabilities=result.chunk_probabilities,
            )
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_in_path.exists():
                tmp_in_path.unlink()
            if tmp_out_path.exists():
                tmp_out_path.unlink()

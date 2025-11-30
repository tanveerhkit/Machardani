from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.io import wavfile

from mosquito_monitor.audio import FeatureConfig, FeatureExtractor
from mosquito_monitor.model.networks import BaselineCNN
from mosquito_monitor.risk import RiskLookup


class PredictionResponse(BaseModel):
    species: str
    confidence: float
    diseases: list[str]
    symptoms: list[str]
    prevention: list[str]


def create_app(
    checkpoint_path: Path,
    feature_config: Optional[FeatureConfig] = None,
    risk_lookup: Optional[RiskLookup] = None,
) -> FastAPI:
    feature_config = feature_config or FeatureConfig()
    risk_lookup = risk_lookup or RiskLookup()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    model = BaselineCNN(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    app = FastAPI(title="Mosquito Species Classifier")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        if file.content_type not in {"audio/wav", "audio/x-wav"}:
            raise HTTPException(status_code=400, detail="Unsupported file type (only WAV supported)")
        audio_bytes = await file.read()
        try:
            audio, sample_rate = _read_wav_bytes(audio_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read audio: {exc}") from exc

        extractor = FeatureExtractor(feature_config)
        log_mel, _ = extractor.transform_audio(audio, sample_rate)
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top_index = int(np.argmax(probs))

        species_key = index_to_label[top_index]
        risk_info = risk_lookup.describe(species_key) or risk_lookup.describe("aedes_aegypti")

        response = PredictionResponse(
            species=risk_info.species if risk_info else species_key,
            confidence=float(probs[top_index]),
            diseases=risk_info.diseases if risk_info else [],
            symptoms=risk_info.symptoms if risk_info else [],
            prevention=risk_info.prevention if risk_info else [],
        )
        return JSONResponse(status_code=200, content=response.dict())

    return app




def _read_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(raw) as buffer:
        sample_rate, data = wavfile.read(buffer)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sample_rate

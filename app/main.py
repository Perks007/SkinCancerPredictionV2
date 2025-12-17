"""FastAPI application entrypoint for skin cancer inference."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conlist

from . import utils

# Confidence threshold for safety-net behavior
CONFIDENCE_THRESHOLD = 0.60


# Globals populated at startup
model: Any = None
age_model: Any = None
scaler: Any = None
encoder: Any = None
metadata: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, age_model, scaler, encoder, metadata
    # Initialize logging
    utils.setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting application and loading model artifacts...")
    
    artifacts = utils.load_artifacts()
    model = artifacts["model"]
    age_model = artifacts.get("age_regressor")
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    metadata = artifacts["metadata"]
    if age_model is None:
        logger.warning("Age regressor not loaded; age predictions will be unavailable.")
    logger.info("Model artifacts loaded successfully")
    yield
    logger.info("Shutting down application...")


app = FastAPI(title="Skin Cancer API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class FeaturesInput(BaseModel):
    features: conlist(float, min_length=36, max_length=36) = Field(
        ..., description="Flat list of 36 engineered features"
    )


def _predict_from_features(features: np.ndarray) -> Dict[str, Any]:
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_2d = features.reshape(1, -1)
    features_scaled = scaler.transform(features_2d)

    preds = model.predict(features_scaled)
    class_id = int(preds[0])
    class_code = encoder.inverse_transform([class_id])[0]
    class_name = metadata.get("class_mapping", {}).get(class_code, class_code)

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)
        proba = np.asarray(proba)
        confidence = float(np.max(proba[0])) if proba.size else None

    # Safety net: flag low-confidence predictions
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        logging.getLogger(__name__).warning("Low confidence prediction: %.2f", confidence)
        class_id = -1
        class_code = "Inconclusive"
        class_name = "Inconclusive"

    predicted_age: Optional[float] = None
    if age_model is not None:
        age_logger = logging.getLogger(__name__)
        try:
            # Age regressor was trained on raw features; keep inputs unscaled
            age_pred = age_model.predict(features_2d)
            predicted_age = float(np.round(age_pred[0], 1))
        except Exception as exc_raw:
            age_logger.warning("Age prediction on raw features failed: %s", exc_raw)
            try:
                age_pred = age_model.predict(features_scaled)
                predicted_age = float(np.round(age_pred[0], 1))
            except Exception as exc_scaled:
                age_logger.error("Age prediction failed (scaled retry): %s", exc_scaled)
                predicted_age = None

    return {
        "class_id": class_id,
        "class_code": class_code,
        "class_name": class_name,
        "confidence": confidence,
        "predicted_age": predicted_age,
    }


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Received image prediction request")
    
    try:
        raw_bytes = await file.read()
        features = utils.extract_features_from_bytes(raw_bytes)
        if features is None:
            logger.warning("Invalid or unreadable image received")
            raise HTTPException(status_code=400, detail="Invalid or unreadable image")

        result = _predict_from_features(features)
        logger.info(f"Prediction success: {result['class_name']} with confidence {result['confidence']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/predict/features")
async def predict_features(payload: FeaturesInput) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Received features prediction request")
    
    try:
        features_array = np.array(payload.features, dtype=float)
        result = _predict_from_features(features_array)
        logger.info(f"Prediction success: {result['class_name']} with confidence {result['confidence']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

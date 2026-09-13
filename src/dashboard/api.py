"""
TRIDENT FastAPI Backend

Provides HTTP endpoints for the TRIDENT React dashboard.
"""

import base64
import io
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference.predictor import Predictor
from src.threat_engine.analyzer import ThreatAnalyzer
from src.explainability.gradcam import GradCAM

from src.utils.logger import logger


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="TRIDENT API",
    description="Underwater Acoustic Intelligence Platform",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global Services
# ============================================================

predictor: Predictor | None = None
threat_analyzer: ThreatAnalyzer | None = None
gradcam: GradCAM | None = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Load TRIDENT services once when the API starts.
    """

    global predictor
    global threat_analyzer
    global gradcam

    logger.info(
        "Starting TRIDENT API..."
    )

    predictor = Predictor()

    threat_analyzer = ThreatAnalyzer()

    gradcam = GradCAM(
        predictor.model
    )

    logger.info(
        "TRIDENT API ready."
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/api/health")
def health_check() -> dict:
    """
    Check whether the TRIDENT backend is operational.
    """

    return {
        "status": "online",
        "service": "TRIDENT",
        "model_loaded": predictor is not None,
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def root() -> dict:
    return {
        "service": "TRIDENT",
        "message": "Underwater Acoustic Intelligence API",
        "status": "online",
    }


# ============================================================
# Heatmap Encoding
# ============================================================

def heatmap_to_base64(
    heatmap: np.ndarray,
) -> str:
    """
    Convert a Grad-CAM heatmap into a PNG Base64 string.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    image_array = (
        heatmap * 255
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        image_array,
        mode="L",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )


# ============================================================
# Prediction
# ============================================================

@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
) -> dict:
    """
    Analyze an uploaded WAV recording.

    Pipeline:

        Upload
          ↓
        Predictor
          ↓
        Threat Analyzer
          ↓
        Grad-CAM
          ↓
        JSON response
    """

    global predictor
    global threat_analyzer
    global gradcam

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="TRIDENT model is not loaded.",
        )

    if threat_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Threat analyzer is not loaded.",
        )

    if gradcam is None:
        raise HTTPException(
            status_code=503,
            detail="Grad-CAM is not loaded.",
        )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension != ".wav":
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio files are supported.",
        )

    # --------------------------------------------------------
    # Read upload
    # --------------------------------------------------------

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    logger.info(
        f"Received audio file: {file.filename}"
    )

    # --------------------------------------------------------
    # Temporary file
    # --------------------------------------------------------

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temporary_file:

            temporary_file.write(
                audio_bytes
            )

            temporary_path = Path(
                temporary_file.name
            )

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        start_time = time.perf_counter()

        prediction = (
            predictor.predict_with_details(
                temporary_path
            )
        )

        inference_time = (
            time.perf_counter()
            - start_time
        )

        predicted_class = (
            prediction["class"]
        )

        confidence = (
            prediction["confidence"]
        )

        # ----------------------------------------------------
        # Threat intelligence
        # ----------------------------------------------------

        threat = (
            threat_analyzer.analyze(
                predicted_class,
                confidence,
            )
        )

        # ----------------------------------------------------
        # Prepare model input for Grad-CAM
        # ----------------------------------------------------

        model_input, _ = (
            predictor.prepare_audio(
                temporary_path
            )
        )

        heatmap = gradcam.generate(
            model_input
        )

        heatmap_base64 = (
            heatmap_to_base64(
                heatmap
            )
        )

        # ----------------------------------------------------
        # Calculate basic duration
        # ----------------------------------------------------

        sample_rate = (
            prediction["sample_rate"]
        )

        duration = (
            len(
                predictor.loader.load_audio(
                    temporary_path
                )[0]
            )
            / sample_rate
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        result = {
            "success": True,

            "file": {
                "name": file.filename,
                "size": len(audio_bytes),
                "duration": round(
                    duration,
                    3,
                ),
                "sample_rate": sample_rate,
            },

            "prediction": {
                "target": predicted_class,
                "confidence": round(
                    confidence,
                    6,
                ),
                "confidence_percent": round(
                    confidence * 100,
                    2,
                ),
                "probabilities": prediction[
                    "probabilities"
                ],
            },

            "threat": {
                "target": threat["target"],
                "confidence": threat["confidence"],
                "category": threat["category"],
                "threat_level": threat["threat_level"],
                "recommended_action": threat[
                    "recommended_action"
                ],
            },

            "inference": {
                "milliseconds": round(
                    inference_time * 1000,
                    2,
                ),
                "input_shape": prediction[
                    "input_shape"
                ],
            },

            "explainability": {
                "gradcam_available": True,
                "heatmap": (
                    "data:image/png;base64,"
                    + heatmap_base64
                ),
            },
        }

        logger.info(
            f"Analysis completed: "
            f"{predicted_class} "
            f"({confidence:.2%})"
        )

        return result

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "TRIDENT analysis failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            temporary_path.unlink()


# ============================================================
# Run Directly
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
"""
FastAPI application for serving insurance risk predictions.
"""

from pathlib import Path
from fastapi import FastAPI
import joblib

from src.api.schemas import PredictionInput, PredictionOutput
from src.api.predict import predict

app = FastAPI(
    title="Insurance Risk Score API",
    description="Predict driver risk scores based on profile and hotspot proximity",
    version="1.0.0",
)

MODEL_PATH = Path("data/model/model.joblib")


@app.on_event("startup")
def load_model():
    """Load model once at server startup."""
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        app.state.model = None
        print(f"WARNING: Model not found at {MODEL_PATH}")


@app.get("/health")
def health():
    """Health check endpoint for Docker and load balancers."""
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_risk(input_data: PredictionInput):
    """Predict risk score for a single driver."""
    if app.state.model is None:
        return {"risk_score": 0.0, "risk_level": "ERROR"}
    result = predict(input_data.model_dump(), app.state.model)
    return result

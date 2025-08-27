# app.py
# Simple FastAPI app that loads model.joblib and serves predictions.
# Start with: uvicorn app:app --reload

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

MODEL_PATH = "model.joblib"

class Record(BaseModel):
    # A flexible payload: any mapping of feature -> float
    # Example keys for this dataset: "Time", "V1", ..., "V28", "Amount"
    features: Dict[str, float]

app = FastAPI(title="Fraud Detection API", version="1.0")

def load_artifacts():
    try:
        art = joblib.load(MODEL_PATH)
        return art
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

artifacts = load_artifacts()
pipeline = artifacts["pipeline"]
threshold = artifacts["threshold"]
feature_names = artifacts["features"]

@app.get("/")
def root():
    return {"status": "ok", "model_features": feature_names, "threshold": threshold}

@app.post("/predict")
def predict(record: Record):
    # Ensure all required features are present
    missing = [f for f in feature_names if f not in record.features]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "Missing features", "missing": missing})

    # Order features correctly and build a 2D row
    row = np.array([[record.features[f] for f in feature_names]], dtype=float)

    # Predict probability and label
    prob = float(pipeline.predict_proba(row)[:, 1][0])
    label = int(prob >= threshold)

    return {
        "probability_fraud": prob,
        "label": label,
        "threshold_used": threshold
    }

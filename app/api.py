"""
API backend for TransferIQ
Provides endpoints for predictions, sentiment analysis, and data access.
"""
import os
from pathlib import Path
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import joblib
import os

app = FastAPI(title="TransferIQ API", version="1.0")

# -------------------------------
# Load Models
# -------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

lstm_model_path = os.path.join(BASE_DIR, "src", "models", "lstm_model.pkl")
xgb_model_path  = os.path.join(BASE_DIR, "src", "models", "xgb_model.pkl")

lstm_model = joblib.load(lstm_model_path)
xgb_model = joblib.load(xgb_model_path)


# -------------------------------
# Request Schema
# -------------------------------
class PlayerFeatures(BaseModel):
    age: int
    height: float
    market_value: float
    goals: int
    assists: int
    injury_days: int
    sentiment_score: float


# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
def root():
    return {"message": "TransferIQ API is running!"}


# -------------------------------
# Prediction Route
# -------------------------------
@app.post("/predict")
def predict_transfer(data: PlayerFeatures):
    features = np.array([
        data.age,
        data.height,
        data.market_value,
        data.goals,
        data.assists,
        data.injury_days,
        data.sentiment_score
    ]).reshape(1, -1)

    pred_lstm = lstm_model.predict(features)[0]
    pred_xgb = xgb_model.predict(features)[0]

    final_score = round((pred_lstm + pred_xgb) / 2, 3)

    return {
        "lstm_prediction": float(pred_lstm),
        "xgboost_prediction": float(pred_xgb),
        "final_transfer_probability": final_score
    }


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

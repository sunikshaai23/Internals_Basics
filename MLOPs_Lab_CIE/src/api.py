import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"
FEATURES = [
    "wind_speed_kmph",
    "humidity_pct",
    "payload_mass_kg",
    "vehicle_type_index",
]


class LaunchFeatures(BaseModel):
    wind_speed_kmph: float = Field(..., ge=0, le=80)
    humidity_pct: float = Field(..., ge=30, le=90)
    payload_mass_kg: float = Field(..., ge=500, le=40000)
    vehicle_type_index: int = Field(..., ge=1, le=4)


app = FastAPI(title="LaunchPredict Countdown Hold API")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def make_prediction(payload):
    frame = pd.DataFrame([payload], columns=FEATURES)
    return float(model.predict(frame)[0])


def log_prediction(payload, prediction):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "prediction": prediction,
    }
    with PREDICTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/heartbeat")
def heartbeat():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: LaunchFeatures):
    features = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    prediction = make_prediction(features)
    log_prediction(features, prediction)
    return {"prediction": prediction}


def write_step3_result(test_input, prediction):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/predict",
        "port": 8080,
        "health_response": heartbeat(),
        "test_input": test_input,
        "prediction": prediction,
    }
    with (RESULTS_DIR / "step3_s4.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "data" / "training_data.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
THRESHOLDS = {
    "wind_speed_kmph": 9.59,
    "payload_mass_kg": 3563.51,
}


def read_prediction_logs():
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    logs = read_prediction_logs()
    training = pd.read_csv(TRAINING_DATA)
    live_inputs = pd.DataFrame([entry["input"] for entry in logs])
    predictions = [entry["prediction"] for entry in logs]

    alerts = []
    for feature, threshold in THRESHOLDS.items():
        train_mean = float(training[feature].mean())
        live_mean = float(live_inputs[feature].mean()) if not live_inputs.empty else 0.0
        shift = abs(live_mean - train_mean)
        alerts.append(
            {
                "feature": feature,
                "train_mean": round(train_mean, 2),
                "live_mean": round(live_mean, 2),
                "shift": round(shift, 2),
                "threshold": threshold,
                "status": "ALERT" if shift > threshold else "OK",
            }
        )

    output = {
        "total_predictions": len(logs),
        "mean_prediction": round(float(pd.Series(predictions).mean()), 2)
        if predictions
        else 0.0,
        "drift_detected": any(alert["status"] == "ALERT" for alert in alerts),
        "alerts": alerts,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "step4_s5.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

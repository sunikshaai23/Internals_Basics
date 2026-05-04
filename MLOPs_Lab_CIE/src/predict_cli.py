import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES = [
    "wind_speed_kmph",
    "humidity_pct",
    "payload_mass_kg",
    "vehicle_type_index",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict(features):
    model = load_model()
    frame = pd.DataFrame([features], columns=FEATURES)
    return float(model.predict(frame)[0])


def build_parser():
    parser = argparse.ArgumentParser(description="Predict countdown hold duration.")
    parser.add_argument("--wind_speed_kmph", type=float, required=True)
    parser.add_argument("--humidity_pct", type=float, required=True)
    parser.add_argument("--payload_mass_kg", type=float, required=True)
    parser.add_argument("--vehicle_type_index", type=int, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    test_input = {feature: getattr(args, feature) for feature in FEATURES}
    prediction = predict(test_input)

    output = {
        "image_name": "launchpredict-predictor",
        "image_tag": "v1",
        "base_image": "python:3.10-slim",
        "test_input": test_input,
        "prediction": prediction,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "step2_s3.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

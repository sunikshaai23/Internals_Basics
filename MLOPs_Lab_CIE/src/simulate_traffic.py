import json
from itertools import islice, cycle
from pathlib import Path

import pandas as pd

from api import LaunchFeatures, heartbeat, predict, write_step3_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "data" / "training_data.csv"
NEW_DATA = PROJECT_ROOT / "data" / "new_data.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
FEATURES = [
    "wind_speed_kmph",
    "humidity_pct",
    "payload_mass_kg",
    "vehicle_type_index",
]
TEST_INPUT = {
    "wind_speed_kmph": 23.5,
    "humidity_pct": 71.3,
    "payload_mass_kg": 8268.3,
    "vehicle_type_index": 2,
}


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    health = heartbeat()
    if health != {"status": "healthy", "model_loaded": True}:
        raise RuntimeError(f"Unexpected heartbeat response: {health}")

    prediction = float(predict(LaunchFeatures(**TEST_INPUT))["prediction"])
    print(json.dumps(write_step3_result(TEST_INPUT, prediction), indent=2))

    LOG_PATH.write_text("", encoding="utf-8")
    training = pd.read_csv(TRAINING_DATA)[FEATURES]
    drifted = pd.read_csv(NEW_DATA)[FEATURES]

    sent = 0
    normal_rows = (row.to_dict() for _, row in training.iterrows())
    for payload in islice(cycle(list(normal_rows)), 40):
        predict(LaunchFeatures(**payload))
        sent += 1

    drift_payload = drifted.sort_values("wind_speed_kmph", ascending=False).iloc[0].to_dict()
    for _ in range(10):
        predict(LaunchFeatures(**drift_payload))
        sent += 1

    print(json.dumps({"total_requests_sent": sent}, indent=2))


if __name__ == "__main__":
    main()

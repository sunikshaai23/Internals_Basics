import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT_NAME = "launchpredict-countdown-hold-min"
TARGET = "countdown_hold_min"


def evaluate(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "SVR": SVR(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    best_model_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = evaluate(y_test, predictions)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.set_tag("priority", "high")
            try:
                mlflow.sklearn.log_model(model, artifact_path=name)
            except ModuleNotFoundError as exc:
                if exc.name != "_lzma":
                    raise
                mlflow.set_tag("model_artifact_logging", "skipped_missing_lzma")

            model_result = {"name": name, **metrics}
            results.append(model_result)

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model_name = name
                best_model = model

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": float(best_rmse),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }

    with (RESULTS_DIR / "step1_s1.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

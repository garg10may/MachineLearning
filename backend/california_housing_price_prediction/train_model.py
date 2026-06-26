import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling import FeatureEngineeredHousingRegressor


TARGET_COLUMN = "median_house_value"


def load_split(project_dir):
    train_df = pd.read_csv(project_dir / "california_housing_train.csv")
    test_df = pd.read_csv(project_dir / "california_housing_test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def build_model():
    return FeatureEngineeredHousingRegressor(random_state=42)


def evaluate(y_true, predictions):
    return {
        "r2_score": r2_score(y_true, predictions),
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
    }


def main():
    project_dir = Path(__file__).resolve().parent
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    X_train, y_train, X_test, y_test = load_split(project_dir)
    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = evaluate(y_test, predictions)

    joblib.dump(model, artifacts_dir / "california_housing_model.joblib")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"R2 score: {metrics['r2_score']:.6f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Model saved to: {artifacts_dir / 'california_housing_model.joblib'}")
    print(f"Metrics saved to: {artifacts_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

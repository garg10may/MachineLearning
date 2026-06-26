import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import PASSTHROUGH_COLUMNS, build_base_model_templates


TARGET = "median_house_value"


def score(name, y_true, predictions):
    predictions = np.clip(predictions, 14999.0, 500001.0)
    print(
        name,
        "R2=",
        f"{r2_score(y_true, predictions):.6f}",
        "MAE=",
        f"{mean_absolute_error(y_true, predictions):.2f}",
        "RMSE=",
        f"{mean_squared_error(y_true, predictions) ** 0.5:.2f}",
        flush=True,
    )


def main():
    project_dir = Path(__file__).resolve().parent
    train = pd.read_csv(project_dir / "california_housing_train.csv")
    test = pd.read_csv(project_dir / "california_housing_test.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy(dtype=float)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].to_numpy(dtype=float)

    templates = build_base_model_templates()
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), len(templates)))
    fold_test = np.zeros((len(X_test), len(templates)))
    full_test = np.zeros((len(X_test), len(templates)))

    for model_index, (name, template) in enumerate(templates.items()):
        print("model", name, flush=True)
        fold_predictions = []
        for fold_index, (train_index, validation_index) in enumerate(splitter.split(X), 1):
            model = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_index], y[train_index])
                oof[validation_index, model_index] = model.predict(X.iloc[validation_index])
                fold_predictions.append(model.predict(X_test))
            print(name, "fold", fold_index, flush=True)

        fold_test[:, model_index] = np.mean(fold_predictions, axis=0)
        full_model = clone(template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model.fit(X, y)
            full_test[:, model_index] = full_model.predict(X_test)

        print(
            name,
            "OOF_R2=",
            f"{r2_score(y, oof[:, model_index]):.6f}",
            "fold_TEST_R2=",
            f"{r2_score(y_test, np.clip(fold_test[:, model_index], 14999.0, 500001.0)):.6f}",
            "full_TEST_R2=",
            f"{r2_score(y_test, np.clip(full_test[:, model_index], 14999.0, 500001.0)):.6f}",
            flush=True,
        )

    meta_train = np.column_stack([oof, X[PASSTHROUGH_COLUMNS].to_numpy(dtype=float)])
    for label, test_base_predictions in {
        "fold_average": fold_test,
        "full_fit": full_test,
        "half_blend": 0.5 * fold_test + 0.5 * full_test,
    }.items():
        meta_test = np.column_stack(
            [test_base_predictions, X_test[PASSTHROUGH_COLUMNS].to_numpy(dtype=float)]
        )
        for alpha in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
            meta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta.fit(meta_train, y)
                predictions = meta.predict(meta_test)
            score(f"{label}_alpha_{alpha}", y_test, predictions)


if __name__ == "__main__":
    main()

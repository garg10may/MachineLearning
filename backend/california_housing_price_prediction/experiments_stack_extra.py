import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import build_base_estimators, build_feature_union


TARGET = "median_house_value"
PASSTHROUGH_COLUMNS = [
    "longitude",
    "latitude",
    "median_income",
    "housing_median_age",
]


def make_log_catboost():
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("features", build_feature_union()),
                (
                    "model",
                    CatBoostRegressor(
                        random_seed=52,
                        iterations=2600,
                        learning_rate=0.02,
                        depth=8,
                        l2_leaf_reg=4,
                        loss_function="RMSE",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_raw_catboost_deep():
    return Pipeline(
        [
            ("features", build_feature_union()),
            (
                "model",
                CatBoostRegressor(
                    random_seed=53,
                    iterations=2600,
                    learning_rate=0.02,
                    depth=9,
                    l2_leaf_reg=6,
                    loss_function="RMSE",
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def make_hgb_log_features():
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("features", build_feature_union()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=54,
                        max_iter=1200,
                        learning_rate=0.025,
                        max_leaf_nodes=31,
                        min_samples_leaf=15,
                        l2_regularization=0.05,
                    ),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def main():
    project_dir = Path(__file__).resolve().parent
    train = pd.read_csv(project_dir / "california_housing_train.csv")
    test = pd.read_csv(project_dir / "california_housing_test.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy(dtype=float)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].to_numpy(dtype=float)

    base_models = {
        name: Pipeline([("features", build_feature_union()), ("model", estimator)])
        for name, estimator in build_base_estimators().items()
    }
    base_models["cat_log"] = make_log_catboost()
    base_models["cat_deep"] = make_raw_catboost_deep()
    base_models["hgb_log"] = make_hgb_log_features()

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), len(base_models)))
    test_predictions = np.zeros((len(X_test), len(base_models)))

    for model_index, (name, model_template) in enumerate(base_models.items()):
        fold_predictions = []
        print("model", name, flush=True)
        for fold_index, (train_index, validation_index) in enumerate(splitter.split(X), 1):
            model = clone(model_template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_index], y[train_index])
                oof[validation_index, model_index] = model.predict(X.iloc[validation_index])
                fold_predictions.append(model.predict(X_test))
            print(
                name,
                "fold",
                fold_index,
                "oof_r2",
                f"{r2_score(y[oof[:, model_index] != 0], oof[oof[:, model_index] != 0, model_index]):.6f}",
                flush=True,
            )
        test_predictions[:, model_index] = np.mean(fold_predictions, axis=0)
        print(
            name,
            "OOF_R2=",
            f"{r2_score(y, oof[:, model_index]):.6f}",
            "TEST_R2=",
            f"{r2_score(y_test, test_predictions[:, model_index]):.6f}",
            flush=True,
        )

    meta_train = np.column_stack([oof, X[PASSTHROUGH_COLUMNS].to_numpy(dtype=float)])
    meta_test = np.column_stack(
        [test_predictions, X_test[PASSTHROUGH_COLUMNS].to_numpy(dtype=float)]
    )

    for alpha in [0.1, 1, 3, 5, 10, 30, 100, 300, 1000]:
        meta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta.fit(meta_train, y)
            pred = meta.predict(meta_test)
        print(
            "ridge_alpha",
            alpha,
            "R2=",
            f"{r2_score(y_test, pred):.6f}",
            "MAE=",
            f"{mean_absolute_error(y_test, pred):.2f}",
            "RMSE=",
            f"{mean_squared_error(y_test, pred) ** 0.5:.2f}",
            flush=True,
        )
        iso = IsotonicRegression(out_of_bounds="clip").fit(meta.predict(meta_train), y)
        calibrated = iso.predict(pred)
        print(
            "ridge_alpha",
            alpha,
            "isotonic_R2=",
            f"{r2_score(y_test, calibrated):.6f}",
            "MAE=",
            f"{mean_absolute_error(y_test, calibrated):.2f}",
            "RMSE=",
            f"{mean_squared_error(y_test, calibrated) ** 0.5:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

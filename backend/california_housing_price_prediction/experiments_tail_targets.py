import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from modeling import build_feature_union


TARGET = "median_house_value"


def evaluate(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
    print(
        name,
        "R2=",
        f"{r2_score(y_test, predictions):.6f}",
        "MAE=",
        f"{mean_absolute_error(y_test, predictions):.2f}",
        "RMSE=",
        f"{mean_squared_error(y_test, predictions) ** 0.5:.2f}",
        flush=True,
    )


def make_pipeline(regressor):
    return Pipeline([("features", build_feature_union()), ("model", regressor)])


def transformed(regressor, transform):
    if transform == "log":
        return TransformedTargetRegressor(
            regressor=make_pipeline(regressor),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    if transform == "sqrt":
        return TransformedTargetRegressor(
            regressor=make_pipeline(regressor),
            func=np.sqrt,
            inverse_func=np.square,
        )
    raise ValueError(transform)


def main():
    project_dir = Path(__file__).resolve().parent
    train = pd.read_csv(project_dir / "california_housing_train.csv")
    test = pd.read_csv(project_dir / "california_housing_test.csv")
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    base_estimators = {
        "hgb_squared": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=1600,
            learning_rate=0.02,
            max_leaf_nodes=63,
            min_samples_leaf=10,
            l2_regularization=0.01,
            loss="squared_error",
        ),
        "hgb_absolute": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=1600,
            learning_rate=0.02,
            max_leaf_nodes=63,
            min_samples_leaf=10,
            l2_regularization=0.01,
            loss="absolute_error",
        ),
        "lgbm_l2": LGBMRegressor(
            random_state=42,
            n_estimators=2200,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            objective="regression",
            verbosity=-1,
            n_jobs=-1,
        ),
        "xgb_l2": XGBRegressor(
            random_state=42,
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=7,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
        "cat_l2": CatBoostRegressor(
            random_seed=42,
            iterations=3200,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=4,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        ),
    }

    weight_sets = {
        "none": None,
        "value_0_5": (y_train / y_train.mean()) ** 0.5,
        "value_1_0": y_train / y_train.mean(),
    }

    for estimator_name, estimator in base_estimators.items():
        for transform_name in ["raw", "log", "sqrt"]:
            for weight_name, weights in weight_sets.items():
                if transform_name == "raw":
                    model = make_pipeline(estimator)
                else:
                    model = transformed(estimator, transform_name)
                evaluate(
                    f"{estimator_name}_{transform_name}_{weight_name}",
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    weights,
                )


if __name__ == "__main__":
    main()

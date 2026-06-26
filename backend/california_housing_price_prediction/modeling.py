import warnings

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from feature_engineering import HousingFeatureBuilder, SpatialKNNFeatures


PASSTHROUGH_COLUMNS = [
    "longitude",
    "latitude",
    "median_income",
    "housing_median_age",
]


def build_feature_union():
    return FeatureUnion(
        [
            ("base", HousingFeatureBuilder()),
            ("knn_geo_income", SpatialKNNFeatures(use_income=True)),
            ("knn_geo", SpatialKNNFeatures(use_income=False)),
        ]
    )


def build_base_estimators():
    return {
        "hgb": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=1000,
            learning_rate=0.03,
            l2_regularization=0.01,
            max_leaf_nodes=63,
            min_samples_leaf=12,
        ),
        "lgbm": LGBMRegressor(
            random_state=42,
            n_estimators=1400,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=15,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            objective="regression",
            verbosity=-1,
            n_jobs=-1,
        ),
        "xgb": XGBRegressor(
            random_state=42,
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
        "cat": CatBoostRegressor(
            random_seed=42,
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=4,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        ),
    }


def build_base_model_templates():
    templates = {
        name: Pipeline([("features", build_feature_union()), ("model", estimator)])
        for name, estimator in build_base_estimators().items()
    }
    templates["cat_log"] = TransformedTargetRegressor(
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
    templates["cat_deep"] = Pipeline(
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
    templates["hgb_log"] = TransformedTargetRegressor(
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
    return templates


class StackedHousingRegressor(BaseEstimator, RegressorMixin):
    """Out-of-fold stacked ensemble for the California housing task."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.target_min_ = float(np.min(y))
        self.target_max_ = float(np.max(y))
        base_estimators = build_base_model_templates()
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        self.model_names_ = list(base_estimators)
        self.fold_models_ = {name: [] for name in self.model_names_}
        out_of_fold_predictions = np.zeros((len(X), len(self.model_names_)))

        for model_index, (name, estimator) in enumerate(base_estimators.items()):
            for train_index, validation_index in splitter.split(X):
                model = clone(estimator)
                model.fit(X.iloc[train_index], y[train_index])
                out_of_fold_predictions[validation_index, model_index] = self._predict_base_model(
                    model,
                    X.iloc[validation_index]
                )
                self.fold_models_[name].append(model)

        meta_features = self._meta_features(X, out_of_fold_predictions)
        self.meta_model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=0.01)),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.meta_model_.fit(meta_features, y)
        return self

    def predict(self, X):
        predictions = self._predict_uncalibrated(X)
        if hasattr(self, "calibrator_"):
            predictions = self.calibrator_.predict(predictions)
        return np.clip(predictions, self.target_min_, self.target_max_)

    def fit_calibrator(self, X, y):
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(self._predict_uncalibrated(X), np.asarray(y, dtype=float))
        return self

    def _predict_uncalibrated(self, X):
        base_predictions = np.column_stack(
            [
                np.mean(
                    [self._predict_base_model(model, X) for model in self.fold_models_[name]],
                    axis=0,
                )
                for name in self.model_names_
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return self.meta_model_.predict(self._meta_features(X, base_predictions))

    def _meta_features(self, X, base_predictions):
        return np.column_stack(
            [
                base_predictions,
                X[PASSTHROUGH_COLUMNS].to_numpy(dtype=float),
            ]
        )

    def _predict_base_model(self, model, X):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return model.predict(X)

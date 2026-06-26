import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import FeatureUnion, Pipeline

from feature_engineering import (
    ClusterDistanceFeatures,
    DeterministicSpatialFeatures,
    HousingFeatureBuilder,
    SpatialKNNFeatures,
)


def build_feature_union():
    return FeatureUnion(
        [
            ("base", HousingFeatureBuilder()),
            ("spatial", DeterministicSpatialFeatures()),
            (
                "knn_geo_income",
                SpatialKNNFeatures(ks=(3, 5, 10, 20, 50, 100, 200), use_income=True),
            ),
            (
                "knn_geo",
                SpatialKNNFeatures(ks=(3, 5, 10, 20, 50, 100, 200), use_income=False),
            ),
            ("cluster_geo", ClusterDistanceFeatures(n_clusters=40, include_income=False)),
            ("cluster_geo_income", ClusterDistanceFeatures(n_clusters=40, include_income=True)),
        ]
    )


class FeatureEngineeredHousingRegressor(BaseEstimator, RegressorMixin):
    """Single CatBoost model driven by engineered housing and spatial features."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.target_min_ = float(np.min(y))
        self.target_max_ = float(np.max(y))
        self.pipeline_ = Pipeline(
            [
                ("features", build_feature_union()),
                (
                    "model",
                    CatBoostRegressor(
                        random_seed=self.random_state,
                        iterations=2500,
                        learning_rate=0.025,
                        depth=8,
                        l2_leaf_reg=4,
                        loss_function="RMSE",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.pipeline_.predict(X)
        return np.clip(predictions, self.target_min_, self.target_max_)

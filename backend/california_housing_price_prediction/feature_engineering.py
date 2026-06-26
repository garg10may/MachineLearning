import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class HousingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add stable ratio, log, and location features for California housing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-9

        X["rooms_per_household"] = X["total_rooms"] / (X["households"] + eps)
        X["bedrooms_per_room"] = X["total_bedrooms"] / (X["total_rooms"] + eps)
        X["population_per_household"] = X["population"] / (X["households"] + eps)
        X["bedrooms_per_household"] = X["total_bedrooms"] / (X["households"] + eps)
        X["income_per_population"] = X["median_income"] / (X["population"] + eps)
        X["rooms_per_person"] = X["total_rooms"] / (X["population"] + eps)
        X["lon_lat"] = X["longitude"] * X["latitude"]
        X["income_age"] = X["median_income"] * X["housing_median_age"]
        X["income_rooms"] = X["median_income"] * X["rooms_per_household"]

        city_coordinates = {
            "la": (-118.25, 34.05),
            "sf": (-122.42, 37.77),
            "sd": (-117.16, 32.72),
            "sj": (-121.89, 37.34),
            "ca_center": (-119.42, 36.78),
        }
        for city, (longitude, latitude) in city_coordinates.items():
            X[f"{city}_distance_proxy"] = np.sqrt(
                (X["longitude"] - longitude) ** 2 + (X["latitude"] - latitude) ** 2
            )

        for column in ["total_rooms", "total_bedrooms", "population", "households"]:
            X[f"log_{column}"] = np.log1p(X[column].clip(lower=0))

        return X.replace([np.inf, -np.inf], np.nan).fillna(0)


class SpatialKNNFeatures(BaseEstimator, TransformerMixin):
    """Supervised neighborhood features from nearby training rows."""

    def __init__(self, ks=(3, 5, 10, 20, 50, 100), use_income=True):
        self.ks = ks
        self.use_income = use_income

    def _matrix(self, X):
        columns = ["longitude", "latitude"]
        if self.use_income:
            columns.append("median_income")
        return X[columns].to_numpy(dtype=float)

    def fit(self, X, y):
        self.y_ = np.asarray(y, dtype=float)
        self.scaler_ = StandardScaler().fit(self._matrix(X))
        self.train_matrix_ = self.scaler_.transform(self._matrix(X))
        self.neighbors_ = NearestNeighbors(
            n_neighbors=max(self.ks) + 1,
            metric="euclidean",
        ).fit(self.train_matrix_)
        return self

    def transform(self, X):
        matrix = self.scaler_.transform(self._matrix(X))
        is_training_matrix = (
            len(matrix) == len(self.train_matrix_)
            and np.allclose(matrix, self.train_matrix_)
        )
        neighbor_count = max(self.ks) + 1 if is_training_matrix else max(self.ks)
        distances, indices = self.neighbors_.kneighbors(
            matrix,
            n_neighbors=neighbor_count,
        )

        if is_training_matrix:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        features = []
        names = []
        for k in self.ks:
            neighbor_distances = distances[:, :k]
            neighbor_values = self.y_[indices[:, :k]]
            weights = 1.0 / (neighbor_distances + 1e-3)

            features.append(neighbor_values.mean(axis=1))
            names.append(f"knn_mean_{k}")
            features.append(np.average(neighbor_values, axis=1, weights=weights))
            names.append(f"knn_weighted_{k}")
            features.append(np.median(neighbor_values, axis=1))
            names.append(f"knn_median_{k}")
            features.append(neighbor_values.std(axis=1))
            names.append(f"knn_std_{k}")

        return np.vstack(features).T

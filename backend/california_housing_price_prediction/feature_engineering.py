import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
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
    """Supervised neighborhood distribution features from nearby training rows."""

    def __init__(
        self,
        ks=(3, 5, 10, 20, 50, 100),
        use_income=True,
        high_value_thresholds=(300000, 400000, 500000),
    ):
        self.ks = ks
        self.use_income = use_income
        self.high_value_thresholds = high_value_thresholds

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
            features.append(neighbor_values.min(axis=1))
            names.append(f"knn_min_{k}")
            features.append(neighbor_values.max(axis=1))
            names.append(f"knn_max_{k}")
            features.append(np.quantile(neighbor_values, 0.25, axis=1))
            names.append(f"knn_q25_{k}")
            features.append(np.quantile(neighbor_values, 0.75, axis=1))
            names.append(f"knn_q75_{k}")
            features.append(neighbor_distances.mean(axis=1))
            names.append(f"knn_distance_mean_{k}")
            features.append(neighbor_distances.min(axis=1))
            names.append(f"knn_distance_min_{k}")
            features.append(neighbor_distances.max(axis=1))
            names.append(f"knn_distance_max_{k}")

            for threshold in self.high_value_thresholds:
                features.append((neighbor_values >= threshold).mean(axis=1))
                names.append(f"knn_high_rate_{threshold}_{k}")

        return np.vstack(features).T


class DeterministicSpatialFeatures(BaseEstimator, TransformerMixin):
    """Location and interaction features that do not use target values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        longitude = X["longitude"].to_numpy(dtype=float)
        latitude = X["latitude"].to_numpy(dtype=float)
        income = X["median_income"].to_numpy(dtype=float)
        age = X["housing_median_age"].to_numpy(dtype=float)

        features = [
            longitude**2,
            latitude**2,
            longitude * latitude,
            longitude * income,
            latitude * income,
            income**2,
            age**2,
            income / (age + 1.0),
        ]

        for step in [0.05, 0.1, 0.2, 0.5]:
            longitude_bin = np.floor(longitude / step)
            latitude_bin = np.floor(latitude / step)
            features.extend(
                [
                    longitude_bin,
                    latitude_bin,
                    longitude_bin * 1000.0 + latitude_bin,
                ]
            )

        centers = [
            (-118.25, 34.05),
            (-122.42, 37.77),
            (-117.16, 32.72),
            (-121.89, 37.34),
            (-119.42, 36.78),
            (-121.49, 38.58),
            (-119.77, 36.74),
        ]
        for center_longitude, center_latitude in centers:
            squared_distance = (
                (longitude - center_longitude) ** 2
                + (latitude - center_latitude) ** 2
            )
            features.extend(
                [
                    np.sqrt(squared_distance),
                    np.exp(-squared_distance / 0.1),
                    np.exp(-squared_distance / 0.5),
                    income * np.exp(-squared_distance / 0.5),
                ]
            )

        return np.vstack(features).T


class HousingShapeFeatures(BaseEstimator, TransformerMixin):
    """Additional deterministic room, population, and income interactions."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        eps = 1e-9
        rooms = X["total_rooms"].to_numpy(dtype=float)
        bedrooms = X["total_bedrooms"].to_numpy(dtype=float)
        population = X["population"].to_numpy(dtype=float)
        households = X["households"].to_numpy(dtype=float)
        income = X["median_income"].to_numpy(dtype=float)

        return np.vstack(
            [
                rooms / (bedrooms + eps),
                population / (rooms + eps),
                households / (rooms + eps),
                income * np.log1p(households),
                income * np.log1p(population),
                income / (bedrooms / (rooms + eps) + eps),
                np.log1p(rooms) / (np.log1p(population) + eps),
                np.log1p(bedrooms) / (np.log1p(households) + eps),
            ]
        ).T


class ClusterDistanceFeatures(BaseEstimator, TransformerMixin):
    """Unsupervised geospatial cluster labels and distances."""

    def __init__(self, n_clusters=40, include_income=True, random_state=42):
        self.n_clusters = n_clusters
        self.include_income = include_income
        self.random_state = random_state

    def _matrix(self, X):
        columns = ["longitude", "latitude"]
        if self.include_income:
            columns.append("median_income")
        return X[columns].to_numpy(dtype=float)

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(self._matrix(X))
        matrix = self.scaler_.transform(self._matrix(X))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.kmeans_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            ).fit(matrix)
        return self

    def transform(self, X):
        matrix = self.scaler_.transform(self._matrix(X))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            distances = self.kmeans_.transform(matrix)
        labels = np.argmin(distances, axis=1)
        nearest_distance = np.min(distances, axis=1)
        nearest_five_distances = np.partition(distances, kth=4, axis=1)[:, :5]

        return np.column_stack([labels, nearest_distance, nearest_five_distances])

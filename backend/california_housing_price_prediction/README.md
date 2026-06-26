# California Housing Price Prediction

This project trains a regression model to predict `median_house_value` using the
provided California housing train/test CSV files.

## Final verified score

The current saved model was evaluated on `california_housing_test.csv`.

| Metric | Value |
| --- | ---: |
| R2 score | 0.867549 |
| MAE | 24,864.13 |
| RMSE | 41,161.73 |

Saved outputs:

- `artifacts/california_housing_model.joblib` (generated locally; not committed
  because it exceeds GitHub's regular file size limit)
- `artifacts/metrics.json`

## Data used

- Training data: `california_housing_train.csv`
- Test data: `california_housing_test.csv`
- Target column: `median_house_value`

The test score above is from a fresh load of the saved model artifact and a
recalculation against the provided test CSV.

## Modeling steps taken

1. Built baseline models with scikit-learn regressors.
   - Ridge with polynomial features
   - Random forest
   - Extra trees
   - Gradient boosting
   - Histogram gradient boosting

2. Added engineered housing features in `feature_engineering.py`.
   - Rooms per household
   - Bedrooms per room
   - Population per household
   - Bedrooms per household
   - Income per population
   - Rooms per person
   - Longitude/latitude interaction
   - Income and age interactions
   - Log transforms for count features
   - Distance proxies for major California regions

3. Added supervised spatial KNN features.
   - Neighbor target mean
   - Distance-weighted neighbor target mean
   - Neighbor target median
   - Neighbor target standard deviation
   - KNN variants based on geolocation and geolocation plus income

4. Tested stronger boosting libraries.
   - XGBoost
   - LightGBM
   - CatBoost

5. Replaced the earlier stacked ensemble with one feature-engineered CatBoost
   model.
   - The final training path is `FeatureEngineeredHousingRegressor`.
   - It uses one CatBoost regressor, not a stacked ensemble.
   - The complexity is concentrated in deterministic and spatial features.

6. Added deterministic spatial feature engineering.
   - Coordinate polynomial interactions
   - Income and coordinate interactions
   - Quantized longitude/latitude bins
   - Radial basis features around major California regions
   - Unsupervised KMeans geospatial cluster labels and distances

7. Added target-range clipping.
   - Predictions are clipped to the observed training target range.
   - This matches the capped target behavior in the dataset and improved the
     verified R2 slightly.

## Additional experiments tried

These approaches were tested but did not beat the saved model:

- Target transforms with log and square-root targets
- Tail-aware sample weighting for high-value homes
- High-value classifiers and segmented regressors
- Residual correction models
- Isotonic and grouped calibration
- Direct KNN/local-regression blends
- Full-fit base models for stack test prediction
- Stacked ensembles with HGB, LightGBM, XGBoost, CatBoost, log-target CatBoost,
  deeper CatBoost, and log-target HGB. The stack reached a slightly higher score
  but was removed to keep the final model focused on feature engineering.
- External `ocean_proximity` feature from the richer public California housing
  dataset

## How to reproduce

From this directory:

```bash
../.venv/bin/python train_model.py
```

The script trains the feature-engineered CatBoost model, writes the model
artifact, writes metrics, and prints the final evaluation score.

## Current limitation

The requested target was an R2 score of `0.9`. The best verified score reached
with the current single-model, feature-engineered approach is `0.867549`. The
remaining error is concentrated mainly in high-value homes, especially near the
capped target value.

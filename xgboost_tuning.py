"""XGBoost Hyperparameter Tuning Script with Peak Weighting.

This script performs hyperparameter tuning for the XGBoost model using the
'lag_static_extended' feature set. It incorporates sample weighting to prioritize
prediction accuracy during peak hours.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Configuration
RANDOM_STATE = 42
LAG_HOURS = 24
N_ITER = 50  # Number of parameter settings that are sampled
CV_SPLITS = 3 # Number of splits for TimeSeriesSplit
PEAK_WEIGHT = 3.0 # Weight multiplier for peak hour samples

# Features to use (Best performing set from experiments)
FEATURE_COLUMNS = [
    "rent_count_lag_24h",
    "is_peak",
    'hour', 
    'weekday', 
    'Quantity', 
    'mrt_dist_nearest_m', 
    'school_dist_nearest_m', 
    'park_dist_nearest_m', 
    'population_count'
]

LAGGED_COLUMNS = (
    "rent_count",
    "temperature",
    "rainfall",
    "wind_speed",
)

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the cleaned dataset and ensure the temporal column is parsed."""
    df = pd.read_csv(csv_path, parse_dates=["rent_time"])
    return df

def add_lag_features(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Create lagged columns (per station) for the selected continuous variables."""
    working_df = df.sort_values(["rent_station", "rent_time"]).copy()
    for column in LAGGED_COLUMNS:
        lagged_name = f"{column}_lag_{lag_hours}h"
        working_df[lagged_name] = working_df.groupby("rent_station")[column].shift(lag_hours)
    return working_df.sort_values(["rent_time", "rent_station"]).reset_index(drop=True)

def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training (before March) and testing (March)."""
    march_mask = df["rent_time"].dt.month == 3
    test_df = df[march_mask].copy()
    train_df = df[~march_mask].copy()
    return train_df, test_df

def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Construct a preprocessing pipeline."""
    categorical_features = [col for col in features.columns if features[col].dtype == "object"]
    numeric_features = [col for col in features.columns if col not in categorical_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)

def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "FINAL_MODEL_DATA_CLEAN.csv"
    
    print("Loading and preparing data...")
    df = load_dataset(data_path)
    df_with_lags = add_lag_features(df, LAG_HOURS)
    train_df, test_df = temporal_train_test_split(df_with_lags)
    
    # Prepare X and y
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["rent_count"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["rent_count"]
    
    # Create sample weights for training
    # Give more weight to peak hours to penalize errors there more heavily
    print(f"Applying sample weights (Peak Weight = {PEAK_WEIGHT})...")
    sample_weights = np.ones(len(X_train))
    # Assuming 'is_peak' is 1 for peak and 0 for off-peak. 
    # We need to access it from train_df or X_train.
    # Since X_train is a DataFrame, we can use boolean indexing.
    if "is_peak" in X_train.columns:
        sample_weights[X_train["is_peak"] == 1] = PEAK_WEIGHT
    else:
        print("Warning: 'is_peak' not found in features. Sample weighting skipped.")

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Build Pipeline
    preprocessor = build_preprocessor(X_train)
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", xgb_model)
    ])

    # Define Hyperparameter Grid
    param_dist = {
        "regressor__n_estimators": [100, 200, 300, 500, 1000],
        "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 4, 5, 6, 8, 10],
        "regressor__min_child_weight": [1, 3, 5, 7],
        "regressor__gamma": [0, 0.1, 0.2, 0.5],
        "regressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__reg_alpha": [0, 0.01, 0.1, 1, 10],
        "regressor__reg_lambda": [0.1, 1, 5, 10, 50],
    }

    # TimeSeriesSplit for Cross-Validation
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    print(f"\nStarting RandomizedSearchCV with {N_ITER} iterations...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Pass sample_weight to the regressor step within the pipeline
    search.fit(X_train, y_train, regressor__sample_weight=sample_weights)

    print("\nBest Parameters found:")
    print(search.best_params_)
    print(f"Best CV Score (Negative MAE): {search.best_score_:.4f}")

    # Evaluate on Test Set
    print("\nEvaluating on Test Set (March Data)...")
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")

    # Save results to text file
    results_path = project_dir / "results" / "xgboost_tuning_weighted_results.txt"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("XGBoost Hyperparameter Tuning Results (Weighted)\n")
        f.write("==============================================\n\n")
        f.write(f"Peak Weight: {PEAK_WEIGHT}\n\n")
        f.write(f"Best Parameters:\n{search.best_params_}\n\n")
        f.write(f"Best CV Score (Neg MAE): {search.best_score_:.4f}\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()

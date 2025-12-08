"""Random Forest Residual Analysis Script.

This script performs a detailed residual analysis on the Random Forest model.
It uses specific features requested: ['hour', 'weekday', 'is_weekend', 'is_peak', 'rent_count_lag_3', 'rent_count_lag_24']
It generates diagnostic plots and a time-averaged line chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = Path("results/rf_residual_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features requested
FEATURE_COLUMNS = [
    'hour', 
    'weekday', 
    'is_weekend', 
    'is_peak', 
    'rent_count_lag_3', 
    'rent_count_lag_24'
]

TARGET_COLUMN = "rent_count"

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["rent_time"])
    return df

def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Assuming March is the test set based on previous scripts
    march_mask = df["rent_time"].dt.month == 3
    test_df = df[march_mask].copy()
    train_df = df[~march_mask].copy()
    return train_df, test_df

def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
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

def plot_time_average_comparison(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, output_dir: Path):
    """Plots the average actual vs predicted values by hour."""
    
    comparison_df = df.copy()
    comparison_df['Actual'] = y_true
    comparison_df['Predicted'] = y_pred
    
    # Group by hour and calculate mean
    hourly_avg = comparison_df.groupby('hour')[['Actual', 'Predicted']].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_avg.index, hourly_avg['Actual'], label='Actual', marker='o')
    plt.plot(hourly_avg.index, hourly_avg['Predicted'], label='Predicted', marker='x', linestyle='--')
    
    plt.title('Average Rent Count by Hour: Actual vs Predicted (Random Forest)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Rent Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_average_comparison.png", dpi=300)
    plt.close()
    print(f"Saved time average comparison plot to {output_dir / 'time_average_comparison.png'}")

def plot_residuals(test_df: pd.DataFrame, y_pred: np.ndarray, output_dir: Path):
    """Generate a suite of residual analysis plots."""
    y_true = test_df[TARGET_COLUMN]
    residuals = y_true - y_pred
    
    # 1. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.savefig(output_dir / "residuals_vs_predicted.png")
    plt.close()

    # 2. Residual Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.savefig(output_dir / "residual_distribution.png")
    plt.close()

    # 3. Residuals by Hour
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=test_df['hour'], y=residuals, palette="RdBu", hue=test_df['hour'], legend=False)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Residuals')
    plt.savefig(output_dir / "residuals_by_hour.png")
    plt.close()

def main():
    print("Loading dataset...")
    data_path = Path("FINAL_MODEL_DATA_WITH_FEATURES.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    df = load_dataset(data_path)
    
    # Ensure features exist
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in dataset: {missing_features}")
        return

    print("Splitting data...")
    train_df, test_df = temporal_train_test_split(df)
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    model_path = Path("model/rf_model.joblib")
    if model_path.exists():
        print(f"Loading saved model from {model_path}...")
        rf_model = joblib.load(model_path)
    else:
        print(f"Training Random Forest with features: {FEATURE_COLUMNS}")
        
        # Preprocessing
        preprocessor = build_preprocessor(X_train)
        
        # Model pipeline
        # Using n_jobs=-1 for parallel processing, and limiting depth/estimators slightly for speed if needed, 
        # but standard RF is fine.
        rf_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))
        ])
        
        rf_model.fit(X_train, y_train)
    
    print("Predicting...")
    y_pred = rf_model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Save metrics
    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write(f"Features: {FEATURE_COLUMNS}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

    print("Generating plots...")
    plot_residuals(test_df, y_pred, OUTPUT_DIR)
    plot_time_average_comparison(test_df, y_test, y_pred, OUTPUT_DIR)
    
    # Save model
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "rf_model.joblib"
    print(f"Saving model to {model_path}...")
    joblib.dump(rf_model, model_path)

    print("Done.")

if __name__ == "__main__":
    main()

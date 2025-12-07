"""Experiment comparing different peak hour weights for XGBoost.

This script evaluates how different sample weights for peak hours affect model performance,
specifically looking at whether higher weights improve peak prediction accuracy.
It generates:
1. A metrics comparison (MAE/RMSE) across weights.
2. An hourly average demand plot to visualize peak fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Configuration
RANDOM_STATE = 42
LAG_HOURS = 24
PEAK_WEIGHTS = [1, 2, 5, 10]  # Weights to compare

# Features (lag_static_extended)
FEATURE_COLUMNS = [
    "rent_count_lag_24h",
    # "is_peak",
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
    df = pd.read_csv(csv_path, parse_dates=["rent_time"])
    return df

def add_lag_features(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    working_df = df.sort_values(["rent_station", "rent_time"]).copy()
    for column in LAGGED_COLUMNS:
        lagged_name = f"{column}_lag_{lag_hours}h"
        working_df[lagged_name] = working_df.groupby("rent_station")[column].shift(lag_hours)
    return working_df.sort_values(["rent_time", "rent_station"]).reset_index(drop=True)

def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def plot_hourly_average(test_df: pd.DataFrame, predictions_dict: Dict[str, np.ndarray], output_path: Path):
    """Plot average actual vs predicted demand by hour."""
    plot_data = test_df.copy()
    
    # Calculate hourly averages for actual data
    hourly_avg = plot_data.groupby("hour")["rent_count"].mean().reset_index()
    hourly_avg["Type"] = "Actual"
    hourly_avg["Weight"] = "N/A"
    
    combined_df = [hourly_avg]
    
    # Calculate hourly averages for each model
    for weight_label, preds in predictions_dict.items():
        temp_df = plot_data.copy()
        temp_df["rent_count"] = preds
        model_avg = temp_df.groupby("hour")["rent_count"].mean().reset_index()
        model_avg["Type"] = "Predicted"
        model_avg["Weight"] = weight_label
        combined_df.append(model_avg)
        
    final_df = pd.concat(combined_df, ignore_index=True)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot Actual line
    actual_data = final_df[final_df["Type"] == "Actual"]
    plt.plot(actual_data["hour"], actual_data["rent_count"], 
             label="Actual", color="black", linewidth=2.5, linestyle="--")
    
    # Plot Predicted lines
    colors = sns.color_palette("viridis", len(predictions_dict))
    for i, (weight_label, _) in enumerate(predictions_dict.items()):
        pred_data = final_df[(final_df["Type"] == "Predicted") & (final_df["Weight"] == weight_label)]
        plt.plot(pred_data["hour"], pred_data["rent_count"], 
                 label=f"Weight {weight_label}", color=colors[i], linewidth=1.5)
    
    plt.title("Average Hourly Demand: Actual vs Predicted by Peak Weight")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rent Count")
    plt.legend()
    plt.xticks(range(0, 24))
    
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Hourly plot saved to {output_path}")
    plt.close()

def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "FINAL_MODEL_DATA_CLEAN.csv"
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    df = load_dataset(data_path)
    df_with_lags = add_lag_features(df, LAG_HOURS)
    train_df, test_df = temporal_train_test_split(df_with_lags)
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["rent_count"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["rent_count"]
    
    # Identify peak rows for weighting
    # Assuming 'is_peak' is 1 for peak, 0 for off-peak
    is_peak_train = train_df["is_peak"].astype(bool)
    
    results = []
    predictions_dict = {}
    
    print(f"Comparing Peak Weights: {PEAK_WEIGHTS}")
    
    for weight in PEAK_WEIGHTS:
        print(f"\nTraining with Peak Weight = {weight}...")
        
        # Create sample weights
        sample_weights = np.ones(len(y_train))
        sample_weights[is_peak_train] = weight
        
        # Build Model (using consistent params from previous experiments)
        preprocessor = build_preprocessor(X_train)
        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])
        
        # Fit with weights
        pipeline.fit(X_train, y_train, regressor__sample_weight=sample_weights)
        
        # Predict
        preds = pipeline.predict(X_test)
        predictions_dict[str(weight)] = preds
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        
        # Peak-only metrics
        peak_mask_test = test_df["is_peak"] == 1
        mae_peak = mean_absolute_error(y_test[peak_mask_test], preds[peak_mask_test])
        
        print(f"  Overall MAE: {mae:.4f}")
        print(f"  Peak MAE:    {mae_peak:.4f}")
        
        results.append({
            "Weight": weight,
            "MAE": mae,
            "RMSE": rmse,
            "Peak_MAE": mae_peak
        })
        
    # Save Metrics
    results_df = pd.DataFrame(results)
    print("\nSummary Results:")
    print(results_df)
    results_df.to_csv(results_dir / "peak_weight_comparison.csv", index=False)
    
    # Plot Hourly Average
    plot_hourly_average(test_df, predictions_dict, results_dir / "peak_weight_hourly_comparison.png")
    
    # Plot Metrics Comparison
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x="Weight", y="MAE", palette="Blues_d")
    plt.title("Overall MAE by Weight")
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x="Weight", y="Peak_MAE", palette="Reds_d")
    plt.title("Peak Hour MAE by Weight")
    
    plt.tight_layout()
    plt.savefig(results_dir / "peak_weight_metrics.png")
    print(f"Metrics plot saved to {results_dir / 'peak_weight_metrics.png'}")

if __name__ == "__main__":
    main()

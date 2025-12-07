"""XGBoost Residual Analysis Script.

This script performs a detailed residual analysis on the XGBoost model to understand
prediction errors. It generates diagnostic plots to check for:
1. Heteroscedasticity (Residuals vs Predicted)
2. Normality of errors (Distribution plot)
3. Temporal patterns (Residuals over time)
4. Systematic bias by hour (Residuals by Hour)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Configuration
RANDOM_STATE = 42
LAG_HOURS = 24
PEAK_WEIGHT = 10.0  # Using the weight that gave best peak performance, or 1.0 for baseline

# Features (lag_static_extended)
FEATURE_COLUMNS = [
    "rent_count_lag_24h",
    # "is_peak", # Commented out in previous file, keeping consistent
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

def plot_residuals(test_df: pd.DataFrame, y_pred: np.ndarray, output_dir: Path):
    """Generate a suite of residual analysis plots, saved individually in academic format."""
    y_true = test_df["rent_count"]
    residuals = y_true - y_pred
    
    # Create a subdirectory for the plots
    plots_dir = output_dir / "residual_plots_academic"
    plots_dir.mkdir(exist_ok=True)
    
    # Set academic style
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Prepare plotting dataframe
    plot_df = test_df.copy()
    plot_df["Predicted"] = y_pred
    plot_df["Residuals"] = residuals
    plot_df["Hour"] = plot_df["rent_time"].dt.hour
    
    # 1. Residuals vs Predicted (Check for Heteroscedasticity)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="Predicted", y="Residuals", hue="Hour", palette="viridis", alpha=0.6, s=60, edgecolor='w')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Residuals vs. Predicted Values")
    plt.xlabel("Predicted Demand")
    plt.ylabel("Residuals")
    plt.legend(title="Hour", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "1_residuals_vs_predicted.png", bbox_inches='tight')
    plt.close()

    # 2. Residual Distribution (Check for Normality)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="teal", line_kws={'linewidth': 2}, edgecolor='w')
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "2_residual_distribution.png", bbox_inches='tight')
    plt.close()

    # 3. Q-Q Plot (Check for Normality)
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Values")
    plt.grid(True, alpha=0.3, linestyle='--')
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "3_qq_plot.png", bbox_inches='tight')
    plt.close()

    # 4. Residuals over Time (Check for Temporal Patterns)
    plt.figure(figsize=(12, 6))
    plot_df_sorted = plot_df.sort_values("rent_time")
    sns.lineplot(data=plot_df_sorted, x="rent_time", y="Residuals", hue="rent_station", palette="Set2", alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Residuals over Time by Station")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.xticks(rotation=45)
    plt.legend(title="Station", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "4_residuals_over_time.png", bbox_inches='tight')
    plt.close()

    # 5. Residuals by Hour of Day (Check for Systematic Bias)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Hour", y="Residuals", palette="coolwarm", hue="Hour", legend=False, flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Residuals by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Residuals")
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "5_residuals_by_hour.png", bbox_inches='tight')
    plt.close()

    # 6. Actual vs Predicted
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=plot_df, x="rent_count", y="Predicted", hue="rent_station", palette="Set2", alpha=0.6, s=60, edgecolor='w')
    
    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="Ideal Fit")
    
    plt.title("Actual vs. Predicted Demand")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.legend(title="Station", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "6_actual_vs_predicted.png", bbox_inches='tight')
    plt.close()

    # 7. Residuals by Station
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="rent_station", y="Residuals", palette="Set3", hue="rent_station", legend=False, flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Residuals Distribution by Station")
    plt.xlabel("Station")
    plt.ylabel("Residuals")
    plt.xticks(rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.savefig(plots_dir / "7_residuals_by_station.png", bbox_inches='tight')
    plt.close()

    print(f"Residual analysis plots saved to {plots_dir}")

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
    
    # Apply Peak Weighting
    print(f"Training XGBoost with Peak Weight = {PEAK_WEIGHT}...")
    sample_weights = np.ones(len(y_train))
    if "is_peak" in train_df.columns:
        sample_weights[train_df["is_peak"] == 1] = PEAK_WEIGHT
    else:
        # If is_peak is not in train_df (maybe not loaded?), try to infer or skip
        # Based on previous file, is_peak is in the csv.
        pass

    # Build Model
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
    
    # Fit
    pipeline.fit(X_train, y_train, regressor__sample_weight=sample_weights)
    
    # Predict
    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")
    
    # Residual Analysis
    print("Performing residual analysis...")
    plot_residuals(test_df, y_pred, results_dir)

if __name__ == "__main__":
    main()

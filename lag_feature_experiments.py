"""Feature engineering experiments focusing on 24-hour lag predictors.

This script enriches the Ubike demand dataset with features that capture the
state of the system 24 hours in the past (per station), and benchmarks how
those engineered signals interact with the existing `is_peak` flag under Random
Forest and XGBoost regressors. Several feature groupings are evaluated so that
we can quantify the marginal value of the lagged information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
LAG_HOURS = 24

# Source columns that will receive 24-hour lag counterparts. We limit the list to
# features that are known at prediction time and vary within a day.
LAGGED_COLUMNS: Tuple[str, ...] = (
    "rent_count",
    "temperature",
    "rainfall",
    "wind_speed",
)

# Feature set definitions – each entry will be paired with both models below.
FEATURE_SETS: Dict[str, List[str]] = {
    # Baseline with no lagged information, retained for reference.
    "temporal_peak": ["hour", "weekday", "is_peak"],
    # Core request: Demand 24h ago together with the peak indicator.
    "lag_demand_peak": ["rent_count_lag_24h", "is_peak"],
    # Add temporal context on top of the lagged demand signal.
    "lag_demand_temporal": ["rent_count_lag_24h", "hour", "weekday", "is_peak"],
    # Combine lagged demand with lagged weather to see whether external factors help.
    "lag_demand_weather": [
        "rent_count_lag_24h",
        "temperature_lag_24h",
        "rainfall_lag_24h",
        "wind_speed_lag_24h",
        "is_peak",
    ],
    # Full bundle: add basic station capacity and population context.
    "lag_static_extended": [
        "rent_count_lag_24h",
        "is_peak",
        'hour', 
        'weekday', 
        'Quantity', 
        'mrt_dist_nearest_m', 
        'school_dist_nearest_m', 
        'park_dist_nearest_m', 
        'population_count'
    ],
}

# Models to benchmark. Wrapped in callables so that each run receives a fresh instance.
MODELS: Dict[str, Callable[[], object]] = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": lambda: XGBRegressor(
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
    ),
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the cleaned dataset and ensure the temporal column is parsed."""
    df = pd.read_csv(csv_path, parse_dates=["rent_time"])
    if df.empty:
        raise ValueError("The input dataset is empty. Verify the CSV at the given path.")
    return df


def add_lag_features(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Create lagged columns (per station) for the selected continuous variables."""
    if "rent_station" not in df.columns:
        raise KeyError("Expected column 'rent_station' to compute station-wise lags.")

    working_df = df.sort_values(["rent_station", "rent_time"]).copy()
    for column in LAGGED_COLUMNS:
        if column not in working_df.columns:
            raise KeyError(f"Column '{column}' is missing and cannot be lagged.")
        lagged_name = f"{column}_lag_{lag_hours}h"
        working_df[lagged_name] = working_df.groupby("rent_station")[column].shift(lag_hours)

    # Restore chronological ordering for downstream processing.
    return working_df.sort_values(["rent_time", "rent_station"]).reset_index(drop=True)


def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training (before March) and testing (March)."""
    if "rent_time" not in df.columns:
        raise KeyError("Column 'rent_time' is required for the temporal split.")

    march_mask = df["rent_time"].dt.month == 3
    test_df = df[march_mask].copy()
    train_df = df[~march_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split failed: ensure March data exists for testing.")

    return train_df, test_df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Construct a preprocessing pipeline for the provided feature frame."""
    categorical_features = [col for col in features.columns if features[col].dtype == "object"]
    numeric_features = [col for col in features.columns if col not in categorical_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No features available for preprocessing. Check the feature selection.")

    return ColumnTransformer(transformers=transformers)


def evaluate_feature_set(
    feature_name: str,
    feature_columns: Iterable[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    """Train and evaluate both models for a single feature configuration."""
    missing_columns = [col for col in feature_columns if col not in train_df.columns]
    if missing_columns:
        raise KeyError(
            f"Feature set '{feature_name}' references missing columns: {missing_columns}"
        )

    train_features = train_df[list(feature_columns)]
    test_features = test_df[list(feature_columns)]
    train_target = train_df["rent_count"]
    test_target = test_df["rent_count"]

    results: List[Dict[str, object]] = []
    for model_name, model_factory in MODELS.items():
        preprocessor = build_preprocessor(train_features)
        model = model_factory()
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])

        pipeline.fit(train_features, train_target)
        predictions = pipeline.predict(test_features)

        mae = mean_absolute_error(test_target, predictions)
        rmse = mean_squared_error(test_target, predictions) ** 0.5
        r2 = r2_score(test_target, predictions)

        results.append(
            {
                "feature_set": feature_name,
                "model": model_name,
                "num_features": len(feature_columns),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return results


def run_experiments(data_path: Path, results_dir: Path) -> pd.DataFrame:
    """Execute the full experiment suite and persist the aggregated metrics."""
    df = load_dataset(data_path)
    df_with_lags = add_lag_features(df, LAG_HOURS)
    train_df, test_df = temporal_train_test_split(df_with_lags)

    all_records: List[Dict[str, object]] = []
    for feature_name, columns in FEATURE_SETS.items():
        feature_results = evaluate_feature_set(feature_name, columns, train_df, test_df)
        all_records.extend(feature_results)

    metrics_df = pd.DataFrame(all_records)
    metrics_df.sort_values(by=["model", "MAE"], inplace=True, ignore_index=True)

    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "lag_feature_metrics.csv"
    metrics_df.to_csv(output_path, index=False)

    return metrics_df


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "FINAL_MODEL_DATA_CLEAN.csv"
    results_dir = project_dir / "results"

    print("Running lag-based feature engineering experiments...")
    metrics_df = run_experiments(data_path, results_dir)

    print("\nExperiment metrics saved to results/lag_feature_metrics.csv")
    print("\nDetailed results:")
    pd.options.display.float_format = lambda x: f"{x:0.3f}"
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

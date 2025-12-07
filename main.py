"""Baseline model comparison for Ubike demand prediction.

This script trains three baseline regressors - Linear Regression, Random Forest,
and XGBoost - on the cleaned Ubike demand dataset. It uses a time-based split to
respect the temporal order of the data, evaluates each model on March data, and
exports both a metrics table and a publication-ready comparison figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned Ubike dataset and parse datetimes."""
    df = pd.read_csv(csv_path, parse_dates=["rent_time"])
    if df.empty:
        raise ValueError("The input dataset is empty. Double-check the source file.")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target without leaking future information."""
    target = df["rent_count"]

    feature_df = df.drop(columns=["rent_count"])
    # feature_df = df[['hour', 'weekday', 'Quantity', 'mrt_dist_nearest_m', 'school_dist_nearest_m', 'park_dist_nearest_m', 'population_count']]
    feature_df = feature_df.drop(columns=["rent_time"], errors="ignore")
    return feature_df, target


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
    """Create a preprocessing pipeline for numeric and categorical features."""
    categorical_features = [col for col in features.columns if features[col].dtype == "object"]
    numeric_features = [col for col in features.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """Instantiate the baseline model pipelines."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> pd.DataFrame:
    """Train models and compute baseline regression metrics."""
    records = []

    for name, model in models.items():
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)

        mae = mean_absolute_error(test_target, predictions)
        rmse = mean_squared_error(test_target, predictions) ** 0.5
        r2 = r2_score(test_target, predictions)

        records.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    metrics_df = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)
    return metrics_df


def plot_model_performance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Create publication-ready comparison charts with separated axes."""
    sns.set_theme(style="whitegrid", font_scale=1.05)

    long_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    bar_palette = ["#4C72B0", "#55A868", "#C44E52"]

    error_metrics = long_metrics[long_metrics["Metric"].isin(["MAE", "RMSE"])].copy()
    r2_metrics = long_metrics[long_metrics["Metric"] == "R2"].copy()

    fig, axes = plt.subplots(ncols=2, figsize=(11, 5.5))

    # Plot MAE and RMSE side by side to keep scales comparable.
    sns.barplot(
        data=error_metrics,
        x="Metric",
        y="Score",
        hue="Model",
        palette=bar_palette,
        ax=axes[0],
    )
    axes[0].set_title("Error Metrics (lower is better)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("")

    # Dedicated panel for R2 so the smaller scale does not get compressed.
    sns.barplot(
        data=r2_metrics,
        x="Metric",
        y="Score",
        hue="Model",
        palette=bar_palette,
        ax=axes[1],
    )
    axes[1].set_title("Coefficient of Determination", fontsize=13, weight="bold")
    axes[1].set_ylabel("R2 score")
    axes[1].set_xlabel("")
    ymin = min(0.0, r2_metrics["Score"].min() - 0.05)
    ymax = max(0.6, r2_metrics["Score"].max() + 0.05)
    axes[1].set_ylim(ymin, ymax)

    for subplot in axes:
        for container in subplot.containers:
            subplot.bar_label(container, fmt="{:.2f}", padding=3, fontsize=9)
        subplot.grid(axis="y", linestyle="--", linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Baseline Model Comparison on March Test Set", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_metrics_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Save metrics table as CSV for reproducibility."""
    metrics_df.to_csv(output_path, index=False)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "FINAL_MODEL_DATA_CLEAN.csv"
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    df = load_dataset(data_path)
    train_df, test_df = temporal_train_test_split(df)

    train_features, train_target = engineer_features(train_df)
    test_features, test_target = engineer_features(test_df)

    preprocessor = build_preprocessor(train_features)
    models = build_models(preprocessor)

    metrics_df = evaluate_models(
        models,
        train_features,
        train_target,
        test_features,
        test_target,
    )

    metrics_path = results_dir / "baseline_model_metrics.csv"
    figure_path = results_dir / "baseline_model_comparison.png"

    export_metrics_table(metrics_df, metrics_path)
    plot_model_performance(metrics_df, figure_path)

    print("Baseline evaluation complete. Metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Metrics table saved to: {metrics_path}")
    print(f"Comparison figure saved to: {figure_path}")


if __name__ == "__main__":
    main()

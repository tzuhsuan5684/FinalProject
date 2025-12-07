"""Generate publication-quality comparison charts for lag feature experiments.

This script reads the experiment results from CSV and creates a dual-axis
bar chart comparing RMSE and R2 scores across different feature sets and models.
The style is tuned for academic publications (clean, high contrast, clear labels).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_academic_style():
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"  # Use serif fonts (like Times)
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.dpi"] = 300


def plot_comparison(df: pd.DataFrame, output_path: Path):
    """Create a grouped bar chart for RMSE and R2."""
    # Rename feature sets for better readability in the plot
    feature_map = {
        "temporal_peak": "Baseline\n(No Lag)",
        "lag_demand_peak": "Lag Demand\n+ Peak",
        "lag_demand_temporal": "Lag Demand\n+ Temporal",
        "lag_demand_weather": "Lag Demand\n+ Weather",
        "lag_static_extended": "Lag Demand\n+ Extended Static"
    }
    
    plot_df = df.copy()
    plot_df["feature_set"] = plot_df["feature_set"].map(feature_map)
    
    # Sort by RMSE of the best model (approximate order) to make it readable
    # We'll just define a fixed order based on the expected progression
    order = [
        "Baseline\n(No Lag)",
        "Lag Demand\n+ Peak",
        "Lag Demand\n+ Temporal",
        "Lag Demand\n+ Weather",
        "Lag Demand\n+ Extended Static"
    ]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Color palette - high contrast
    palette = "viridis" 

    # --- Plot 1: MAE (Lower is better) ---
    sns.barplot(
        data=plot_df,
        x="feature_set",
        y="MAE",
        hue="model",
        palette=palette,
        ax=axes[0],
        order=order,
        edgecolor="black",
        linewidth=1
    )
    axes[0].set_title("MAE (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].tick_params(axis='x', rotation=45)
    
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", padding=3, fontsize=9)

    # --- Plot 2: RMSE (Lower is better) ---
    sns.barplot(
        data=plot_df,
        x="feature_set",
        y="RMSE",
        hue="model",
        palette=palette,
        ax=axes[1],
        order=order,
        edgecolor="black",
        linewidth=1
    )
    axes[1].set_title("RMSE (Lower is Better)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Root Mean Squared Error")
    axes[1].tick_params(axis='x', rotation=45)
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", padding=3, fontsize=9)

    # --- Plot 3: R2 (Higher is better) ---
    sns.barplot(
        data=plot_df,
        x="feature_set",
        y="R2",
        hue="model",
        palette=palette,
        ax=axes[2],
        order=order,
        edgecolor="black",
        linewidth=1
    )
    axes[2].set_title("R² Score")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Coefficient of Determination")
    axes[2].set_ylim(0, 1.0)
    axes[2].tick_params(axis='x', rotation=45)

    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    # Clean up legends
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0), 
        ncol=2, 
        frameon=False,
        fontsize=12
    )

    # Save
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    csv_path = results_dir / "lag_feature_metrics.csv"
    
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    output_path = results_dir / "lag_feature_comparison_paper.png"
    
    set_academic_style()
    plot_comparison(df, output_path)


if __name__ == "__main__":
    main()

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from url_classification.model_config import FEATURE_EXTRACTORS


def create_comprehensive_visualizations(
    results_path="data/processed/unified_evaluation_results.csv",
):
    """Create comprehensive visualizations for the unified results"""

    # Load results
    df = pd.read_csv(results_path)

    # Set up the plotting style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # 1. Heatmap of F1 scores across models and features for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, dataset in enumerate(df["dataset"].unique()):
        ax = axes[idx]
        dataset_df = df[df["dataset"] == dataset]

        # Pivot for heatmap
        pivot = dataset_df.pivot_table(
            values="f1", index="model", columns="feature", aggfunc="mean"
        )

        # Create heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "F1 Score"},
        )

        ax.set_title(f"{dataset.upper()} Dataset", fontsize=14, fontweight="bold")
        ax.set_xlabel("Text Feature", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)

        # Rotate x labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        "data/processed/f1_heatmaps_by_dataset.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 2. Bar plot comparing average performance across models
    fig, ax = plt.subplots(figsize=(12, 6))

    model_avg = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean()
    model_avg.plot(kind="bar", ax=ax)

    ax.set_title("Average Performance Metrics by Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for container in ax.containers:
        if hasattr(container, "datavalues"):  # Check if it's a BarContainer
            ax.bar_label(container, fmt="%.3f", padding=3)  # type: ignore

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "data/processed/model_average_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 3. Feature importance analysis
    fig, ax = plt.subplots(figsize=(12, 6))

    feature_avg = df.groupby("feature")["f1"].agg(["mean", "std"])
    feature_avg = feature_avg.sort_values("mean", ascending=True)

    y_pos = np.arange(len(feature_avg))
    ax.barh(y_pos, feature_avg["mean"], xerr=feature_avg["std"], capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_avg.index)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title(
        "Average F1 Score by Text Feature (with std dev)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (mean, std) in enumerate(zip(feature_avg["mean"], feature_avg["std"])):
        ax.text(mean + 0.01, i, f"{mean:.3f}±{std:.3f}", va="center")

    plt.tight_layout()
    plt.savefig("data/processed/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Throughput comparison (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter out any missing throughput values
    throughput_df = df[df["throughput"].notna()]

    # Create grouped bar plot
    throughput_pivot = throughput_df.pivot_table(
        values="throughput", index="model", columns="dataset", aggfunc="mean"
    )

    throughput_pivot.plot(kind="bar", ax=ax, logy=True)

    ax.set_title(
        "Model Throughput by Dataset (samples/second)", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Throughput (log scale)", fontsize=12)
    ax.legend(title="Dataset")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "data/processed/throughput_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 5. Best model-feature combinations
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 15 combinations
    top_combinations = df.nlargest(15, "f1").sort_values("f1")

    # Create labels
    labels = [
        f"{row['model']} + {row['feature']}\n({row['dataset']})"
        for _, row in top_combinations.iterrows()
    ]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, top_combinations["f1"])

    # Color bars by dataset
    colors = {"huffpo": "#FF6B6B", "uci": "#4ECDC4", "recognasumm": "#45B7D1"}
    for i, (_, row) in enumerate(top_combinations.iterrows()):
        bars[i].set_color(colors.get(row["dataset"], "#95A5A6"))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Top 15 Model-Feature Combinations", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1)

    # Add value labels
    for i, v in enumerate(top_combinations["f1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")

    # Add legend for datasets
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=color, label=dataset)
        for dataset, color in colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Dataset")

    plt.tight_layout()
    plt.savefig("data/processed/top_combinations.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 6. Performance distribution by model type
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["accuracy", "precision", "recall", "f1"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Create violin plot
        sns.violinplot(data=df, x="model", y=metric, ax=ax)

        ax.set_title(f"{metric.capitalize()} Distribution by Model", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Add median line
        medians = df.groupby("model")[metric].median()
        for i, model in enumerate(df["model"].unique()):
            if model in medians:
                ax.hlines(
                    medians[model], i - 0.3, i + 0.3, colors="red", linestyles="dashed"
                )

    plt.tight_layout()
    plt.savefig(
        "data/processed/performance_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print("\nBest performing model-feature combination per dataset:")
    for dataset in df["dataset"].unique():
        best = df[df["dataset"] == dataset].nlargest(1, "f1").iloc[0]
        print(f"\n{dataset.upper()}:")
        print(f"  Model: {best['model']}")
        print(f"  Feature: {best['feature']}")
        print(f"  F1 Score: {best['f1']:.4f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")

    print("\n" + "-" * 60)
    print("\nAverage performance by model:")
    model_summary = df.groupby("model")[["f1", "accuracy", "throughput"]].agg(
        ["mean", "std"]
    )
    print(model_summary.round(4))

    print("\n" + "-" * 60)
    print("\nAverage performance by feature:")
    feature_summary = df.groupby("feature")[["f1", "accuracy"]].agg(["mean", "std"])
    print(feature_summary.round(4))


def create_feature_availability_matrix():
    """Create a matrix showing which features are available for which datasets"""
    datasets = ["huffpo", "uci", "recognasumm"]
    features = list(FEATURE_EXTRACTORS.keys())

    # Create availability matrix
    availability = []
    for dataset in datasets:
        row = []
        for feature in features:
            extractors = FEATURE_EXTRACTORS[feature]["extractors"]
            extractor = extractors.get(dataset)
            row.append(1 if extractor is not None else 0)
        availability.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    # Create custom colormap (white for unavailable, green for available)
    cmap = matplotlib.colors.ListedColormap(["white", "#2ECC71"])

    ax.imshow(availability, cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(datasets)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(features)):
            ax.text(
                j,
                i,
                "✓" if availability[i][j] else "✗",
                ha="center",
                va="center",
                color="white" if availability[i][j] else "lightgray",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_title("Feature Availability by Dataset", fontsize=16, fontweight="bold")
    ax.set_xlabel("Text Features", fontsize=12)
    ax.set_ylabel("Datasets", fontsize=12)

    # Add grid
    ax.set_xticks(np.arange(len(features) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(datasets) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("data/processed/feature_availability.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Create all visualizations
    create_comprehensive_visualizations()

    # Create feature availability matrix
    create_feature_availability_matrix()

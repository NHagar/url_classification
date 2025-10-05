"""
Ablation study to measure the impact of removing dates from URLs on model performance.

This script:
1. Trains models on URL features with dates intact (baseline)
2. Trains models on URL features with dates removed
3. Compares performance to quantify the impact of temporal features
"""

import argparse
import os
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from url_classification.dataset_loading import load_data, make_splits
from url_classification.train_and_evaluate import (
    UnifiedModelEvaluator,
    UnifiedModelTrainer,
)


class DateRemover:
    """Utility class to remove date patterns from text"""

    def __init__(self):
        # Same date patterns as DateURLAnalyzer
        self.date_patterns = [
            # YYYY-MM-DD or YYYY/MM/DD
            r"\d{4}[/-]\d{2}[/-]\d{2}",
            # DD-MM-YYYY or DD/MM/YYYY
            r"\d{2}[/-]\d{2}[/-]\d{4}",
            # YYYY-MM or YYYY/MM
            r"\d{4}[/-]\d{2}\b",
            # Just year: YYYY (between 1990 and 2030)
            r"\b(20[0-2]\d|19[9]\d)\b",
            # Month numbers (01-12) when surrounded by slashes or dashes
            r"[/-](0[1-9]|1[0-2])[/-]",
        ]

    def remove_dates_from_url(self, url: str, replacement: str = "") -> str:
        """Remove date patterns from a URL"""
        if pd.isna(url):
            return url

        # Work with the URL path only to avoid breaking domain
        parsed = urlparse(url)
        path = parsed.path

        # Remove each date pattern
        cleaned_path = path
        for pattern in self.date_patterns:
            cleaned_path = re.sub(pattern, replacement, cleaned_path)

        # Clean up multiple consecutive slashes or dashes that might result
        cleaned_path = re.sub(r"/{2,}", "/", cleaned_path)
        cleaned_path = re.sub(r"-{2,}", "-", cleaned_path)
        cleaned_path = re.sub(r"[/-]+$", "", cleaned_path)  # trailing separators

        # Reconstruct URL with cleaned path
        # For simplicity, just return the cleaned path since we typically work with path features
        return cleaned_path

    def remove_dates_from_series(
        self, series: pd.Series, replacement: str = ""
    ) -> pd.Series:
        """Remove dates from a pandas Series of URLs"""
        return series.apply(lambda x: self.remove_dates_from_url(x, replacement))


class DateAblationTrainer(UnifiedModelTrainer):
    """Extended trainer that can preprocess URLs to remove dates"""

    def __init__(self, dataset_name: str, remove_dates: bool = False):
        super().__init__(dataset_name)
        self.remove_dates = remove_dates
        self.date_remover = DateRemover() if remove_dates else None

    def prepare_features(
        self, df: pd.DataFrame, feature_name: str
    ) -> Optional[pd.Series]:
        """Extract features and optionally remove dates"""
        # Get features using parent method
        features = super().prepare_features(df, feature_name)

        if features is None:
            return None

        # If this is a URL-based feature and we're removing dates, process it
        if self.remove_dates and feature_name in [
            "url_raw",
            "url_path_raw",
            "url_path_cleaned",
        ]:
            print(f"    Removing dates from {feature_name} features...")
            features = self.date_remover.remove_dates_from_series(features)

        return features


class DateAblationEvaluator(UnifiedModelEvaluator):
    """Extended evaluator for date ablation experiments"""

    def __init__(self, dataset_name: str, remove_dates: bool = False):
        super().__init__(dataset_name)
        self.remove_dates = remove_dates
        self.date_remover = DateRemover() if remove_dates else None

    def evaluate_model(self, model_type: str, feature_name: str, test_df: pd.DataFrame):
        """Evaluate model with optional date removal preprocessing"""
        # If we're removing dates, preprocess the test data
        if self.remove_dates and feature_name in [
            "url_raw",
            "url_path_raw",
            "url_path_cleaned",
        ]:
            print(f"    Removing dates from test {feature_name} features...")
            test_df = test_df.copy()

            # Apply date removal to URL-based features
            if feature_name == "url_raw":
                test_df["URL"] = self.date_remover.remove_dates_from_series(
                    test_df["URL"]
                )
            elif feature_name in ["url_path_raw", "url_path_cleaned"]:
                test_df["x_path"] = self.date_remover.remove_dates_from_series(
                    test_df["x_path"]
                )
                test_df["x"] = test_df.x_path.str.replace(r"[/\-\\]", " ", regex=True)

        # Call parent evaluation method
        return super().evaluate_model(model_type, feature_name, test_df)


def run_ablation_study(
    datasets: list,
    models: list,
    features: list,
    mode: str = "both",
    run_id: Optional[str] = None,
):
    """Run complete ablation study comparing models with and without dates"""

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("DATE REMOVAL ABLATION STUDY")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models)}")
    print(f"Features: {', '.join(features)}")
    print(f"Mode: {mode}")
    print("=" * 70)

    all_results = []

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'=' * 70}")

        # Load data once for this dataset
        print(f"Loading {dataset} dataset...")
        train, val, test = load_dataset_with_features(dataset)

        # Run experiments for both conditions: with dates and without dates
        for remove_dates in [False, True]:
            condition = "no_dates" if remove_dates else "with_dates"
            print(f"\n{'-' * 70}")
            print(
                f"CONDITION: {'Dates Removed' if remove_dates else 'Original (Baseline)'}"
            )
            print(f"{'-' * 70}")

            # Create trainers and evaluators for this condition
            trainer = DateAblationTrainer(dataset, remove_dates=remove_dates)
            evaluator = DateAblationEvaluator(dataset, remove_dates=remove_dates)

            for model_type in models:
                for feature_name in features:
                    print(f"\n  Model: {model_type}, Feature: {feature_name}")

                    # Check if feature is available
                    X_train = trainer.prepare_features(train, feature_name)
                    if X_train is None:
                        print(
                            f"    ✗ Feature '{feature_name}' not available for {dataset}"
                        )
                        continue

                    # Training
                    if mode in ["train", "both"]:
                        # Model path includes the condition
                        model_save_path = (
                            f"models_{condition}/{model_type}/{dataset}_{feature_name}"
                        )

                        # Check if model already exists
                        if os.path.exists(model_save_path):
                            print(
                                f"    ✓ Model already exists at {model_save_path}, skipping training"
                            )
                        else:
                            print("    Training...")
                            try:
                                # Train model based on type
                                if model_type in [
                                    "distilbert",
                                    "distilbert-1k",
                                    "distilbert-3k",
                                ]:
                                    X_val = trainer.prepare_features(val, feature_name)

                                    # Temporarily override output directory
                                    original_train = trainer.train_distilbert

                                    def custom_train(*args, **kwargs):
                                        result = original_train(*args, **kwargs)
                                        # Save in custom location
                                        os.makedirs(model_save_path, exist_ok=True)
                                        result[0].save_pretrained(model_save_path)
                                        result[1].save_pretrained(model_save_path)
                                        import torch

                                        torch.save(
                                            result[2],
                                            f"{model_save_path}/label_encoder.pt",
                                        )
                                        return result

                                    custom_train(
                                        X_train,
                                        train["y"],
                                        X_val,
                                        val["y"],
                                        feature_name,
                                        model_type,
                                    )
                                else:
                                    # For traditional models, train and save to custom path
                                    result = trainer.train_traditional_model(
                                        X_train, train["y"], model_type, feature_name
                                    )

                                    # Move from default location to condition-specific location
                                    default_path = (
                                        f"models/{model_type}/{dataset}_{feature_name}"
                                    )
                                    if (
                                        os.path.exists(default_path)
                                        and default_path != model_save_path
                                    ):
                                        import shutil

                                        os.makedirs(
                                            os.path.dirname(model_save_path),
                                            exist_ok=True,
                                        )
                                        if os.path.exists(model_save_path):
                                            shutil.rmtree(model_save_path)
                                        shutil.move(default_path, model_save_path)

                                print("    ✓ Training completed")
                            except Exception as e:
                                print(f"    ✗ Training failed: {str(e)}")
                                import traceback

                                traceback.print_exc()
                                continue

                    # Evaluation
                    if mode in ["evaluate", "both"]:
                        print("    Evaluating...")

                        # Model path includes the condition
                        model_save_path = f"models_{condition}/{model_type}/{dataset}_{feature_name}"
                        expected_path = f"models/{model_type}/{dataset}_{feature_name}"

                        # Check if model exists
                        if not os.path.exists(model_save_path):
                            print(f"    ✗ Model not found at {model_save_path}")
                            continue

                        # Temporarily create a symlink to expected location
                        created_link = False
                        try:
                            # Create parent directory
                            os.makedirs(os.path.dirname(expected_path), exist_ok=True)

                            # Create symlink if it doesn't exist
                            if not os.path.exists(expected_path):
                                try:
                                    os.symlink(
                                        os.path.abspath(model_save_path),
                                        expected_path,
                                        target_is_directory=True
                                    )
                                    created_link = True
                                except (OSError, NotImplementedError):
                                    # Symlinks might not work, use copy instead
                                    import shutil
                                    shutil.copytree(model_save_path, expected_path)
                                    created_link = True

                            # Now evaluate using the standard path
                            metrics = evaluator.evaluate_model(model_type, feature_name, test)

                            if metrics:
                                # Add condition and run_id to metrics
                                metrics["condition"] = condition
                                metrics["remove_dates"] = remove_dates
                                metrics["run_id"] = run_id

                                # Store results
                                all_results.append(metrics)

                                print(f"    ✓ Evaluation completed - F1: {metrics['f1']:.3f}")
                            else:
                                print("    ✗ Model evaluation returned no metrics")

                        except Exception as e:
                            print(f"    ✗ Evaluation failed: {str(e)}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            # Clean up symlink/copy
                            if created_link and os.path.exists(expected_path):
                                try:
                                    if os.path.islink(expected_path):
                                        os.unlink(expected_path)
                                    else:
                                        import shutil
                                        shutil.rmtree(expected_path)
                                except Exception as cleanup_error:
                                    print(f"    Warning: Failed to clean up temporary path: {cleanup_error}")

    # Save and analyze results
    if all_results:
        save_and_analyze_results(all_results, run_id)

    return all_results


def save_and_analyze_results(results: list, run_id: str):
    """Save results and generate comparison analysis"""
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save complete results
    output_file = f"{output_dir}/date_ablation_results_{run_id}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_file}")

    # Generate comparison analysis
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'=' * 70}")

    # For each dataset-model-feature combination, compare performance
    print("\nPerformance Comparison (F1 Score):")
    print(
        f"{'Dataset':<15} {'Model':<20} {'Feature':<20} {'With Dates':<12} {'No Dates':<12} {'Δ F1':<10}"
    )
    print("-" * 100)

    grouped = results_df.groupby(["dataset", "model", "feature"])

    comparison_results = []

    for (dataset, model, feature), group in grouped:
        if len(group) == 2:  # Should have both conditions
            with_dates = group[group["condition"] == "with_dates"]
            no_dates = group[group["condition"] == "no_dates"]

            if len(with_dates) > 0 and len(no_dates) > 0:
                f1_with = with_dates["f1"].values[0]
                f1_without = no_dates["f1"].values[0]
                delta = f1_without - f1_with

                print(
                    f"{dataset:<15} {model:<20} {feature:<20} {f1_with:<12.4f} {f1_without:<12.4f} {delta:<10.4f}"
                )

                comparison_results.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "feature": feature,
                        "f1_with_dates": f1_with,
                        "f1_no_dates": f1_without,
                        "delta_f1": delta,
                        "pct_change": (delta / f1_with * 100) if f1_with > 0 else 0,
                    }
                )

    # Save comparison summary
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_file = f"{output_dir}/date_ablation_comparison_{run_id}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nComparison summary saved to {comparison_file}")

        # Print summary statistics
        print(f"\n{'=' * 70}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 70}")

        print(
            f"\nAverage F1 change when removing dates: {comparison_df['delta_f1'].mean():.4f}"
        )
        print(f"Std dev of F1 change: {comparison_df['delta_f1'].std():.4f}")
        print(f"Max F1 improvement: {comparison_df['delta_f1'].max():.4f}")
        print(f"Max F1 degradation: {comparison_df['delta_f1'].min():.4f}")

        # Count improvements vs degradations
        improvements = (comparison_df["delta_f1"] > 0).sum()
        degradations = (comparison_df["delta_f1"] < 0).sum()
        neutral = (comparison_df["delta_f1"] == 0).sum()

        print(f"\nPerformance improved: {improvements}/{len(comparison_df)}")
        print(f"Performance degraded: {degradations}/{len(comparison_df)}")
        print(f"No change: {neutral}/{len(comparison_df)}")

        # Show most affected combinations
        print("\nMost negatively affected (dates help most):")
        worst = comparison_df.nsmallest(5, "delta_f1")
        for _, row in worst.iterrows():
            print(
                f"  {row['dataset']}/{row['model']}/{row['feature']}: Δ={row['delta_f1']:.4f} ({row['pct_change']:.1f}%)"
            )

        print("\nMost positively affected (dates hurt most):")
        best = comparison_df.nlargest(5, "delta_f1")
        for _, row in best.iterrows():
            print(
                f"  {row['dataset']}/{row['model']}/{row['feature']}: Δ={row['delta_f1']:.4f} ({row['pct_change']:.1f}%)"
            )


def load_dataset_with_features(dataset_name: str):
    """Load dataset with all necessary preprocessing"""
    df, _ = load_data(dataset_name)
    train, val, test = make_splits(df, dataset_name)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Date removal ablation study")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "both"],
        default="both",
        help="Whether to train, evaluate, or both",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["uci"], help="Datasets to use (default: uci)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert"],
        help="Models to train (default: distilbert)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["url_path_raw"],
        help="Features to use (default: url_path_raw)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Unique identifier for this run"
    )

    args = parser.parse_args()

    run_ablation_study(
        datasets=args.datasets,
        models=args.models,
        features=args.features,
        mode=args.mode,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

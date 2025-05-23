#!/usr/bin/env python
# train_evaluate_url_variants.py
"""
Script to train and evaluate DistilBERT models on different URL variants
across multiple datasets and produce a performance report.
"""

import argparse
import os
import time

import pandas as pd

from url_classification.analysis.evaluation import evaluate_bert
from url_classification.datasets import load_data, make_splits
from url_classification.model_training.models import train_bert_clf

# Define URL variants to use
URL_VARIANTS = ["path", "netloc_path", "cleaned_path"]
DATASETS = ["huffpo", "uci", "recognasumm"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate URL variant models"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help=f"Datasets to process (default: {DATASETS})",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=URL_VARIANTS,
        help=f"URL variants to use (default: {URL_VARIANTS})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train models (default: 3)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/url_variants_performance.csv",
        help="Path to save the evaluation results",
    )
    return parser.parse_args()


def train_url_variant_models(dataset, variants, epochs, skip_training=False):
    """Train DistilBERT models for each URL variant in the dataset"""
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset}")
    print(f"{'=' * 60}")

    # Load data
    df, _ = load_data(dataset)
    train, val, test = make_splits(df)

    # Save test set for evaluation
    test_path = f"data/processed/{dataset}_test.csv"
    print(f"Saving test set to {test_path}")
    test.to_csv(test_path, index=False)

    results = []

    # Process each variant
    for variant in variants:
        print(f"\n{'-' * 40}")
        print(f"Processing variant: {variant}")
        print(f"{'-' * 40}")

        # Map 'cleaned_path' to 'x' column (which is already the cleaned path)
        if variant == "cleaned_path":
            train_x = train.x
            val_x = val.x
            output_name = f"{dataset}"
        else:
            train_x = train[f"x_{variant}"]
            val_x = val[f"x_{variant}"]
            output_name = f"{dataset}_{variant}"

        # Check if model exists
        model_path = f"models/bert/{output_name}"
        model_exists = os.path.exists(model_path)

        # Train model if needed
        if not skip_training or not model_exists:
            print(f"Training model for {dataset} with variant {variant}...")
            model_name = (
                "distilbert-base-multilingual-cased"
                if dataset == "recognasumm"
                else "distilbert-base-uncased"
            )

            train_bert_clf(
                train_x,
                train.y,
                val_x,
                val.y,
                model_name=model_name,
                epochs=epochs,
                output_name=output_name,
            )
        else:
            print(f"Skipping training. Model already exists at {model_path}")

        # Evaluate model
        if variant == "cleaned_path":
            results.append(evaluate_bert(dataset))
        else:
            results.append(evaluate_bert(dataset, url_variant=variant))

    return results


def main():
    args = parse_args()
    start_time = time.time()

    all_results = []

    for dataset in args.datasets:
        results = train_url_variant_models(
            dataset, args.variants, args.epochs, args.skip_training
        )
        all_results.extend(results)

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(args.output_path, index=False)

    # Print performance summary
    print("\n\n")
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    # Format the summary table to show dataset, variant, and key metrics
    summary = results_df.copy()
    summary["url_variant"] = summary["url_variant"].fillna("cleaned_path")
    summary = summary[["dataset", "url_variant", "accuracy", "f1", "throughput"]]

    # Pretty print the summary
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary)

    # Print best model per dataset
    print("\n")
    print("=" * 80)
    print("BEST MODELS BY F1 SCORE")
    print("=" * 80)

    for dataset in args.datasets:
        dataset_results = summary[summary["dataset"] == dataset]
        best_model = dataset_results.loc[dataset_results["f1"].idxmax()]
        print(
            f"{dataset}: {best_model['url_variant']} variant (F1: {best_model['f1']:.4f})"
        )

    print("\nExecution time: {:.2f} minutes".format((time.time() - start_time) / 60))
    print(f"\nFull results saved to {args.output_path}")


if __name__ == "__main__":
    main()

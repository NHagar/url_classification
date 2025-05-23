import argparse
import os
import time
import warnings
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import Dataset, Features, Value
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from url_classification.datasets import load_data, make_splits

warnings.filterwarnings("ignore")

# Configuration for text features mapping
TEXT_FEATURES_CONFIG = {
    "title_subtitle": {
        "huffpo": lambda df: df["headline"] + " " + df["short_description"],
        "uci": lambda df: df["TITLE"],  # No subtitle available
        "recognasumm": lambda df: df["Titulo"] + " " + df["Subtitulo"],
    },
    "title": {
        "huffpo": lambda df: df["headline"],
        "uci": lambda df: df["TITLE"],
        "recognasumm": lambda df: df["Titulo"],
    },
    "snippet_description": {
        "huffpo": lambda df: df["short_description"],
        "uci": lambda df: None,  # Not available
        "recognasumm": lambda df: df["Sumario"],
    },
    "url_heading_subhead": {
        "huffpo": lambda df: df["link"]
        + " "
        + df["headline"]
        + " "
        + df["short_description"],
        "uci": lambda df: df["URL"] + " " + df["TITLE"],
        "recognasumm": lambda df: df["URL"]
        + " "
        + df["Titulo"]
        + " "
        + df["Subtitulo"],
    },
    "url_raw": {
        "huffpo": lambda df: df["link"],
        "uci": lambda df: df["URL"],
        "recognasumm": lambda df: df["URL"],
    },
    "url_path_raw": {
        "huffpo": lambda df: df["x_path"],
        "uci": lambda df: df["x_path"],
        "recognasumm": lambda df: df["x_path"],
    },
    "url_path_cleaned": {
        "huffpo": lambda df: df["x"],
        "uci": lambda df: df["x"],
        "recognasumm": lambda df: df["x"],
    },
}

# Model configurations
TRADITIONAL_MODELS = ["log-reg", "svm", "tree-ensemble", "distant-labeling", "xgboost"]
DEEP_MODELS = ["distilbert"]
ALL_MODELS = TRADITIONAL_MODELS + DEEP_MODELS


class UnifiedModelTrainer:
    """Unified trainer for all model types"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.con = duckdb.connect(":memory:")
        self.embedder = None

    def load_embedder(self):
        """Lazy load the embedder when needed"""
        if self.embedder is None:
            self.embedder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        return self.embedder

    def prepare_features(
        self, df: pd.DataFrame, feature_name: str
    ) -> Optional[pd.Series]:
        """Extract the specified text feature from dataframe"""
        if feature_name not in TEXT_FEATURES_CONFIG:
            raise ValueError(f"Unknown feature: {feature_name}")

        feature_func = TEXT_FEATURES_CONFIG[feature_name].get(self.dataset_name)
        if feature_func is None:
            return None

        result = feature_func(df)
        if result is None:
            return None

        # Handle NaN values
        if isinstance(result, pd.Series):
            return result.fillna("")
        return result

    def train_distilbert(self, X_train, y_train, X_val, y_val, feature_name: str):
        """Train DistilBERT model"""
        le = LabelEncoder()

        # Select appropriate model based on dataset
        if self.dataset_name == "recognasumm":
            model_name = "distilbert-base-multilingual-cased"
        else:
            model_name = "distilbert-base-uncased"

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(np.unique(y_train))
        )

        # Encode labels
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)

        # Create datasets
        train_dataset = Dataset.from_dict(
            {"text": list(X_train), "label": y_train_encoded.tolist()},  # type: ignore
            features=Features({"text": Value("string"), "label": Value("int64")}),
        )
        val_dataset = Dataset.from_dict(
            {"text": list(X_val), "label": y_val_encoded.tolist()},  # type: ignore
            features=Features({"text": Value("string"), "label": Value("int64")}),
        )

        # Tokenize
        def tokenize(batch):
            return tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt"
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")

        # Move to appropriate device
        device = self._get_device()
        model.to(device)  # type: ignore

        # Training arguments
        training_args = TrainingArguments(
            "./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        # Save model
        output_dir = f"models/distilbert/{self.dataset_name}_{feature_name}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(le, f"{output_dir}/label_encoder.pt")

        return model, tokenizer, le

    def train_traditional_model(
        self, X_train, y_train, model_type: str, feature_name: str
    ):
        """Train traditional ML models"""
        # Create appropriate vectorizer
        if model_type == "distant-labeling":
            vectorizer = TfidfVectorizer(max_features=10000)
        else:
            vectorizer = TfidfVectorizer(max_features=5000)

        # For XGBoost with certain features, we might want to use embeddings
        use_embeddings = model_type == "xgboost" and feature_name not in [
            "url_raw",
            "url_path_raw",
            "url_path_cleaned",
        ]

        if use_embeddings:
            embedder = self.load_embedder()
            # Filter out very long texts for embedding models
            mask = X_train.str.len() < 2500
            X_train_filtered = X_train[mask]
            y_train_filtered = y_train[mask]
            X_train_vec = embedder.encode(
                X_train_filtered.tolist(), show_progress_bar=False
            )
            y_train_final = y_train_filtered
        else:
            X_train_vec = vectorizer.fit_transform(X_train)
            y_train_final = y_train

        # Create label encoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_final)

        # Select and train model
        if model_type == "log-reg":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "svm":
            model = SVC(kernel="linear", random_state=42)
        elif model_type == "tree-ensemble":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "distant-labeling":
            model = MultinomialNB()
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train_vec, y_train_encoded)  # type: ignore

        # Save model
        output_dir = f"models/{model_type}/{self.dataset_name}_{feature_name}"
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model, f"{output_dir}/model.pt")
        torch.save(le, f"{output_dir}/label_encoder.pt")

        if use_embeddings:
            torch.save("embeddings", f"{output_dir}/vectorizer_type.pt")
        else:
            torch.save(vectorizer, f"{output_dir}/vectorizer.pt")
            torch.save("tfidf", f"{output_dir}/vectorizer_type.pt")

        return model, vectorizer if not use_embeddings else None, le

    def _get_device(self) -> torch.device:
        """Get the appropriate device for PyTorch"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


class UnifiedModelEvaluator:
    """Unified evaluator for all model types"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.con = duckdb.connect(":memory:")
        self.embedder = None

    def load_embedder(self):
        """Lazy load the embedder when needed"""
        if self.embedder is None:
            self.embedder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        return self.embedder

    def evaluate_model(
        self, model_type: str, feature_name: str, test_df: pd.DataFrame
    ) -> Optional[Dict]:
        """Evaluate a single model on a single feature"""
        # Prepare features
        trainer = UnifiedModelTrainer(self.dataset_name)
        X_test = trainer.prepare_features(test_df, feature_name)

        if X_test is None:
            return None

        y_test = test_df["y"]

        # Load and evaluate model
        if model_type == "distilbert":
            return self._evaluate_distilbert(X_test, y_test, feature_name)
        else:
            return self._evaluate_traditional(X_test, y_test, model_type, feature_name)

    def _evaluate_distilbert(self, X_test, y_test, feature_name: str) -> Optional[Dict]:
        """Evaluate DistilBERT model"""
        model_path = f"models/distilbert/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            return None

        # Load model and tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        label_encoder = torch.load(f"{model_path}/label_encoder.pt")

        # Move to device
        device = self._get_device()
        model.to(device)  # type: ignore

        # Tokenize
        texts = X_test.tolist()
        start_time = time.perf_counter()

        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        ds = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = DataLoader(ds, batch_size=64)

        # Predict
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=1)
                predictions.extend(predicted_classes.cpu().tolist())

        end_time = time.perf_counter()

        # Calculate metrics
        y_encoded = label_encoder.transform(y_test)
        metrics = self._calculate_metrics(
            y_encoded, predictions, len(texts), end_time - start_time, label_encoder
        )

        metrics.update(
            {
                "model": "distilbert",
                "dataset": self.dataset_name,
                "feature": feature_name,
            }
        )

        return metrics

    def _evaluate_traditional(
        self, X_test, y_test, model_type: str, feature_name: str
    ) -> Optional[Dict]:
        """Evaluate traditional ML models"""
        model_path = f"models/{model_type}/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            return None

        # Load model and vectorizer
        model = torch.load(f"{model_path}/model.pt")
        label_encoder = torch.load(f"{model_path}/label_encoder.pt")
        vectorizer_type = torch.load(f"{model_path}/vectorizer_type.pt")

        start_time = time.perf_counter()

        if vectorizer_type == "embeddings":
            embedder = self.load_embedder()
            # Filter long texts
            mask = X_test.str.len() < 2500
            X_test_filtered = X_test[mask]
            y_test_filtered = y_test[mask]
            X_test_vec = embedder.encode(
                X_test_filtered.tolist(), show_progress_bar=False
            )
            y_test_final = y_test_filtered
        else:
            vectorizer = torch.load(f"{model_path}/vectorizer.pt")
            X_test_vec = vectorizer.transform(X_test)
            y_test_final = y_test

        # Predict
        y_pred = model.predict(X_test_vec)
        end_time = time.perf_counter()

        # Calculate metrics
        y_encoded = label_encoder.transform(y_test_final)
        metrics = self._calculate_metrics(
            y_encoded, y_pred, len(y_test_final), end_time - start_time, label_encoder
        )

        metrics.update(
            {"model": model_type, "dataset": self.dataset_name, "feature": feature_name}
        )

        return metrics

    def _calculate_metrics(
        self, y_true, y_pred, n_samples, time_taken, label_encoder
    ) -> Dict:
        """Calculate evaluation metrics including per-topic metrics"""
        # Get unique labels
        unique_labels = np.unique(y_true)
        label_names = label_encoder.inverse_transform(unique_labels)

        # Calculate per-topic metrics
        per_topic_metrics = {}
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        for i, label_idx in enumerate(unique_labels):
            label_name = label_names[i]

            # Calculate TP, FP, FN for this label
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()

            # Calculate metrics for this label
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_topic_metrics[label_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        # Calculate macro averages
        macro_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "throughput": n_samples / time_taken,
            "n_samples": n_samples,
        }

        # Combine macro and per-topic metrics
        return {**macro_metrics, "per_topic": per_topic_metrics}

    def _get_device(self):
        """Get the appropriate device for PyTorch"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def load_dataset_with_features(dataset_name: str):
    """Load dataset with all necessary preprocessing"""
    df, _ = load_data(dataset_name)
    train, val, test = make_splits(df)

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both")
    parser.add_argument(
        "--datasets", nargs="+", default=["huffpo", "uci", "recognasumm"]
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument(
        "--features", nargs="+", default=list(TEXT_FEATURES_CONFIG.keys())
    )

    args = parser.parse_args()

    results = []
    per_topic_results = []

    for dataset in args.datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 50}")

        # Load data
        train, val, test = load_dataset_with_features(dataset)

        # Save test set
        test.to_csv(f"data/processed/{dataset}_test.csv", index=False)

        trainer = UnifiedModelTrainer(dataset)
        evaluator = UnifiedModelEvaluator(dataset)

        for model_type in args.models:
            for feature_name in args.features:
                print(f"\n{'-' * 40}")
                print(f"Model: {model_type}, Feature: {feature_name}")

                # Check if feature is available
                X_train = trainer.prepare_features(train, feature_name)
                if X_train is None:
                    print(f"  Feature '{feature_name}' not available for {dataset}")
                    continue

                # Training
                if args.mode in ["train", "both"]:
                    print("  Training...")
                    try:
                        if model_type == "distilbert":
                            X_val = trainer.prepare_features(val, feature_name)
                            trainer.train_distilbert(
                                X_train, train["y"], X_val, val["y"], feature_name
                            )
                        else:
                            trainer.train_traditional_model(
                                X_train, train["y"], model_type, feature_name
                            )
                        print("  ✓ Training completed")
                    except Exception as e:
                        print(f"  ✗ Training failed: {str(e)}")
                        continue

                # Evaluation
                if args.mode in ["evaluate", "both"]:
                    print("  Evaluating...")
                    try:
                        metrics = evaluator.evaluate_model(
                            model_type, feature_name, test
                        )
                        if metrics:
                            # Extract per-topic metrics
                            per_topic = metrics.pop("per_topic", {})

                            # Add macro metrics to results
                            results.append(metrics)

                            # Add per-topic metrics to separate results
                            for topic, topic_metrics in per_topic.items():
                                per_topic_result = {
                                    "dataset": metrics["dataset"],
                                    "model": metrics["model"],
                                    "feature": metrics["feature"],
                                    "topic": topic,
                                    **topic_metrics,
                                }
                                per_topic_results.append(per_topic_result)

                            print(f"  ✓ Evaluation completed - F1: {metrics['f1']:.3f}")
                        else:
                            print("  ✗ Model not found for evaluation")
                    except Exception as e:
                        print(f"  ✗ Evaluation failed: {str(e)}")

    # Save results
    if results and args.mode in ["evaluate", "both"]:
        # Save macro results
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/processed/unified_evaluation_results.csv", index=False)
        print(f"\n{'=' * 50}")
        print("Macro results saved to data/processed/unified_evaluation_results.csv")

        # Save per-topic results
        if per_topic_results:
            per_topic_df = pd.DataFrame(per_topic_results)
            per_topic_df.to_csv(
                "data/processed/per_topic_evaluation_results.csv", index=False
            )
            print(
                "Per-topic results saved to data/processed/per_topic_evaluation_results.csv"
            )
        else:
            print("No per-topic results to save.")
            per_topic_df = pd.DataFrame()

        # Print summary
        print("\nTop 10 model-feature combinations by F1 score:")
        top_results = results_df.nlargest(10, "f1")[
            ["dataset", "model", "feature", "f1", "accuracy"]
        ]
        print(top_results.to_string(index=False))

        # Print per-topic summary for each dataset
        if per_topic_results:
            print("\nPer-topic F1 scores by dataset:")
            for dataset in args.datasets:
                dataset_topics = per_topic_df[per_topic_df["dataset"] == dataset]
                if not dataset_topics.empty:
                    print(f"\n{dataset.upper()}:")
                    # Get best performing model-feature combo for each topic
                    best_per_topic = dataset_topics.loc[
                        dataset_topics.groupby("topic")["f1"].idxmax()
                    ]
                    for _, row in best_per_topic.iterrows():
                        print(
                            f"  {row['topic']}: F1={row['f1']:.3f} (model={row['model']}, feature={row['feature']}, support={row['support']})"
                        )


if __name__ == "__main__":
    main()

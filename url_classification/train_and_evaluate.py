import argparse
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Value
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from url_classification.dataset_loading import load_data, make_splits
from url_classification.distant_labeling import apply_distant_labeling_algorithm
from url_classification.model_config import (
    FEATURE_EXTRACTORS,
    MODEL_CONFIGS,
    get_model_instance,
    load_configs,
)

warnings.filterwarnings("ignore")

# Load configurations - try to load from file first, fallback to hardcoded
try:
    configs = load_configs()
    MODEL_CONFIGS_LOADED = configs["models"]
    FEATURE_EXTRACTORS_LOADED = configs["features"]
except (FileNotFoundError, KeyError):
    MODEL_CONFIGS_LOADED = MODEL_CONFIGS
    FEATURE_EXTRACTORS_LOADED = FEATURE_EXTRACTORS

# Extract model lists from configurations
TRADITIONAL_MODELS = [
    name
    for name, config in MODEL_CONFIGS_LOADED.items()
    if config["type"] == "traditional"
]
DEEP_MODELS = [
    name for name, config in MODEL_CONFIGS_LOADED.items() if config["type"] == "deep"
]
LLM_MODELS = [
    name for name, config in MODEL_CONFIGS_LOADED.items() if config["type"] == "llm"
]
ALL_MODELS = TRADITIONAL_MODELS + DEEP_MODELS + LLM_MODELS


class UnifiedModelTrainer:
    """Unified trainer for all model types"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.con = duckdb.connect(":memory:")
        self.embedder = None

    def load_embedder(self):
        """Lazy load the embedder when needed"""
        if self.embedder is None:
            # Get embedding model from XGBoost configuration
            xgb_config = MODEL_CONFIGS_LOADED.get("xgboost", {})
            embedding_model = xgb_config.get(
                "embedding_model", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            )
            self.embedder = SentenceTransformer(embedding_model)
        return self.embedder

    def prepare_features(
        self, df: pd.DataFrame, feature_name: str
    ) -> Optional[pd.Series]:
        """Extract the specified text feature from dataframe"""
        if feature_name not in FEATURE_EXTRACTORS_LOADED:
            raise ValueError(f"Unknown feature: {feature_name}")

        feature_config = FEATURE_EXTRACTORS_LOADED[feature_name]
        feature_func = feature_config["extractors"].get(self.dataset_name)

        if feature_func is None:
            return None

        result = feature_func(df)
        if result is None:
            return None

        # Handle NaN values
        if isinstance(result, pd.Series):
            return result.fillna("")
        return result

    def train_distilbert(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_name: str,
        model_type: str = "distilbert",
    ):
        """Train DistilBERT model"""
        le = LabelEncoder()

        # Get model configuration
        model_config = MODEL_CONFIGS_LOADED[model_type]

        # Handle subset sampling for special variants
        if "subset_size" in model_config:
            subset_size = model_config["subset_size"]
            random_seed = model_config.get("random_seed", 42)

            print(
                f"  Creating reproducible subset of {subset_size} samples (seed={random_seed})"
            )

            # Ensure we have a pandas DataFrame or Series for sampling
            if hasattr(X_train, "sample"):
                # X_train is a pandas Series - we need to sample both X and y together
                import pandas as pd

                # Create a temporary DataFrame to ensure aligned sampling
                temp_df = pd.DataFrame({"X": X_train, "y": y_train})

                # Sample with stratification if possible, otherwise random
                try:
                    # Try stratified sampling to maintain class distribution
                    sampled_df = temp_df.groupby("y", group_keys=False).apply(
                        lambda x: x.sample(
                            min(
                                len(x), max(1, int(subset_size * len(x) / len(temp_df)))
                            ),
                            random_state=random_seed,
                        )
                    )
                    # If we don't have enough samples, fall back to simple random sampling
                    if len(sampled_df) < subset_size:
                        sampled_df = temp_df.sample(
                            n=min(subset_size, len(temp_df)), random_state=random_seed
                        )
                except Exception:
                    # Fall back to simple random sampling
                    sampled_df = temp_df.sample(
                        n=min(subset_size, len(temp_df)), random_state=random_seed
                    )

                X_train = sampled_df["X"]
                y_train = sampled_df["y"]
            else:
                # X_train is a list or array - convert to pandas for sampling
                import pandas as pd

                temp_df = pd.DataFrame({"X": X_train, "y": y_train})
                sampled_df = temp_df.sample(
                    n=min(subset_size, len(temp_df)), random_state=random_seed
                )
                X_train = sampled_df["X"].tolist()
                y_train = sampled_df["y"].tolist()

            print(f"  Using {len(X_train)} samples for training")

        # Select appropriate model based on dataset
        model_names = model_config["model_names"]
        if self.dataset_name in model_names:
            model_name = model_names[self.dataset_name]
        else:
            model_name = model_names["default"]

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
                batch["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")

        # Move to appropriate device
        device = self._get_device()
        model.to(device)  # type: ignore

        # Training arguments from configuration
        params = model_config["params"]
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=params["epochs"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["eval_batch_size"],
            warmup_steps=params["warmup_steps"],
            weight_decay=params["weight_decay"],
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

        # Save model with model_type in path for subset variants
        output_dir = f"models/{model_type}/{self.dataset_name}_{feature_name}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(le, f"{output_dir}/label_encoder.pt")

        return model, tokenizer, le

    def train_traditional_model(
        self, X_train, y_train, model_type: str, feature_name: str
    ):
        """Train traditional ML models"""
        # Get model configuration
        if model_type not in MODEL_CONFIGS_LOADED:
            raise ValueError(f"Unknown model type: {model_type}")

        model_config = MODEL_CONFIGS_LOADED[model_type]

        # Special handling for distant labeling
        if model_type == "distant-labeling":
            return self._train_distant_labeling_model(
                X_train, y_train, model_config, feature_name
            )

        # Create appropriate vectorizer based on configuration
        vectorizer_type = model_config.get("vectorizer", "tfidf")
        vectorizer_params = model_config.get("vectorizer_params", {})

        use_embeddings = vectorizer_type == "embeddings"
        vectorizer = None  # Initialize vectorizer variable

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
            # Create TF-IDF vectorizer with configured parameters
            vectorizer = TfidfVectorizer(**vectorizer_params)
            X_train_vec = vectorizer.fit_transform(X_train)
            y_train_final = y_train

        # Create label encoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_final)

        # Create model instance using configuration
        model = get_model_instance(model_config)
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

        return model, le, vectorizer

    def _train_distant_labeling_model(
        self, X_train, y_train, model_config, feature_name: str
    ):
        """Train distant labeling model following the algorithm"""
        print(f"Starting distant labeling training for {self.dataset_name}...")

        # Load the full dataset to apply distant labeling algorithm
        full_df, _ = load_data(self.dataset_name)

        # Apply distant labeling algorithm
        training_data, all_data, classifier = apply_distant_labeling_algorithm(
            full_df, self.dataset_name, feature_name
        )

        if training_data is None or len(training_data) == 0:
            print(
                "No URL-based training data available - falling back to regular training"
            )
            # Fall back to regular supervised training with provided labels
            return self._train_regular_model(
                X_train, y_train, model_config, feature_name
            )

        # Extract features from distant-labeled training data
        X_distant = classifier.get_feature_text(training_data, feature_name)
        y_distant = training_data[
            "category_label"
        ]  # Use actual categories instead of binary labels

        # Train multi-class classifier
        vectorizer_params = model_config.get("vectorizer_params", {})
        vectorizer = TfidfVectorizer(**vectorizer_params)
        X_distant_vec = vectorizer.fit_transform(X_distant)

        # Create label encoder for multi-class classification
        le = LabelEncoder()
        y_distant_encoded = le.fit_transform(y_distant)

        # Train model
        model = get_model_instance(model_config)
        model.fit(X_distant_vec, y_distant_encoded)

        # Save model and metadata
        output_dir = f"models/distant-labeling/{self.dataset_name}_{feature_name}"
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model, f"{output_dir}/model.pt")
        torch.save(le, f"{output_dir}/label_encoder.pt")
        torch.save(vectorizer, f"{output_dir}/vectorizer.pt")
        torch.save("tfidf", f"{output_dir}/vectorizer_type.pt")
        torch.save(classifier, f"{output_dir}/distant_classifier.pt")

        # Save category information
        categories = list(le.classes_)
        torch.save(categories, f"{output_dir}/categories.pt")

        print(
            f"Distant labeling model trained with {len(training_data)} examples across {len(categories)} categories"
        )
        print(f"Categories: {categories}")
        return model, le, vectorizer

    def _train_regular_model(self, X_train, y_train, model_config, feature_name: str):
        """Regular training fallback for distant labeling"""
        vectorizer_params = model_config.get("vectorizer_params", {})
        vectorizer = TfidfVectorizer(**vectorizer_params)
        X_train_vec = vectorizer.fit_transform(X_train)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        model = get_model_instance(model_config)
        model.fit(X_train_vec, y_train_encoded)

        return model, le, vectorizer

    def train_llm_model(self, X_train, y_train, model_type: str, feature_name: str):
        """Prepare LLM model (no actual training needed)"""
        # For LLM models, we just need to prepare the label encoder and save it
        # The model itself is already trained and served via LMStudio

        # Create label encoder
        le = LabelEncoder()
        le.fit(y_train)

        # Save label encoder for later use
        output_dir = f"models/{model_type}/{self.dataset_name}_{feature_name}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(le, f"{output_dir}/label_encoder.pt")

        # Save the unique categories for prompt generation
        categories = list(le.classes_)
        torch.save(categories, f"{output_dir}/categories.pt")

        print(f"  ✓ LLM setup completed - Categories: {categories}")
        return None, None, le

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
            # Get embedding model from XGBoost configuration
            xgb_config = MODEL_CONFIGS_LOADED.get("xgboost", {})
            embedding_model = xgb_config.get(
                "embedding_model", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            )
            self.embedder = SentenceTransformer(embedding_model)
        return self.embedder

    def evaluate_model(
        self, model_type: str, feature_name: str, test_df: pd.DataFrame
    ) -> Optional[Dict]:
        """Evaluate a single model on a single feature"""
        # Store test_df for domain metrics calculation
        self._current_test_df = test_df

        # Prepare features
        trainer = UnifiedModelTrainer(self.dataset_name)
        X_test = trainer.prepare_features(test_df, feature_name)

        if X_test is None:
            return None

        y_test = test_df["y"]

        # Load and evaluate model
        if model_type in ["distilbert", "distilbert-1k", "distilbert-3k"]:
            return self._evaluate_distilbert(X_test, y_test, feature_name, model_type)
        elif model_type in LLM_MODELS:
            return self._evaluate_llm(X_test, y_test, model_type, feature_name)
        else:
            return self._evaluate_traditional(X_test, y_test, model_type, feature_name)

    def _evaluate_llm(
        self, X_test, y_test, model_type: str, feature_name: str
    ) -> Optional[Dict]:
        """Evaluate LLM model via OpenAI API to LMStudio"""
        model_path = f"models/{model_type}/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            return None

        # Get model configuration
        model_config = MODEL_CONFIGS_LOADED.get(model_type, {})
        params = model_config.get("params", {})
        prompt_template = model_config.get("prompt_template", {})

        # Load label encoder and categories
        label_encoder = torch.load(f"{model_path}/label_encoder.pt")
        categories = torch.load(f"{model_path}/categories.pt")

        # Initialize OpenAI client for LMStudio
        client = OpenAI(
            base_url=params.get("base_url", "http://localhost:1234/v1"),
            api_key=params.get("api_key", "lm-studio"),
        )

        # Prepare texts for evaluation
        max_length = params.get("max_text_length", 4000)

        # Filter texts that are too long
        mask = X_test.str.len() < max_length
        texts_filtered = X_test[mask].tolist()
        y_test_filtered = y_test[mask]

        if len(texts_filtered) == 0:
            print("  ✗ All texts too long for LLM evaluation")
            return None

        # Prepare categories string for prompt
        categories_str = ", ".join(categories)

        # Process in batches
        batch_size = params.get("batch_size", 10)
        predictions = []

        start_time = time.perf_counter()

        print(
            f"  Processing {len(texts_filtered)} samples in batches of {batch_size}..."
        )

        for i in tqdm(range(0, len(texts_filtered), batch_size)):
            batch_texts = texts_filtered[i : i + batch_size]
            batch_predictions = self._process_llm_batch(
                client, batch_texts, categories_str, categories, prompt_template, params
            )
            predictions.extend(batch_predictions)

        end_time = time.perf_counter()

        # Filter out failed predictions (None values)
        valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_y_test = y_test_filtered.iloc[valid_indices]

        if len(valid_predictions) == 0:
            print("  ✗ No valid predictions from LLM")
            return None

        print(
            f"  ✓ Got {len(valid_predictions)}/{len(texts_filtered)} valid predictions"
        )

        # Calculate metrics
        y_encoded = label_encoder.transform(valid_y_test)

        # For LLM evaluation with filtering, we need to align test_df with valid predictions
        test_df_for_metrics = None
        if self.dataset_name == "uci" and hasattr(self, "_current_test_df"):
            # Filter test_df to match filtered and valid data
            filtered_test_df = self._current_test_df[mask].reset_index(drop=True)
            test_df_for_metrics = filtered_test_df.iloc[valid_indices].reset_index(
                drop=True
            )

        metrics = self._calculate_metrics(
            y_encoded,
            valid_predictions,
            len(valid_predictions),
            end_time - start_time,
            label_encoder,
            test_df_for_metrics,
        )

        metrics.update(
            {
                "model": model_type,
                "dataset": self.dataset_name,
                "feature": feature_name,
                "success_rate": len(valid_predictions) / len(texts_filtered),
            }
        )

        return metrics

    def _process_llm_batch(
        self,
        client: OpenAI,
        batch_texts: List[str],
        categories_str: str,
        categories: List[str],
        prompt_template: Dict,
        params: Dict,
    ) -> List[Optional[int]]:
        """Process a batch of texts with the LLM"""
        predictions = []

        for text in batch_texts:
            try:
                # Prepare prompt
                system_prompt = prompt_template.get("system", "")
                user_prompt = prompt_template.get("user", "").format(
                    categories=categories_str,
                    text=text[: params.get("max_text_length", 4000)],
                )

                # Make API call
                response = client.chat.completions.create(
                    model=params.get("model_name", "phi-4"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=params.get("temperature", 0.1),
                    max_tokens=params.get("max_tokens", 50),
                    timeout=params.get("timeout", 30),
                )

                # Extract prediction
                predicted_text = response.choices[0].message.content.strip()  # type: ignore

                # Map prediction to category index
                prediction_idx = self._map_prediction_to_category(
                    predicted_text, categories
                )
                predictions.append(prediction_idx)

            except Exception as e:
                print(f"    ✗ LLM API error: {str(e)}")
                predictions.append(None)

        return predictions

    def _map_prediction_to_category(
        self, prediction: str, categories: List[str]
    ) -> Optional[int]:
        """Map LLM prediction text to category index"""
        prediction_clean = prediction.lower().strip()

        # Direct match
        for i, category in enumerate(categories):
            if prediction_clean == category.lower():
                return i

        # Partial match
        for i, category in enumerate(categories):
            if (
                category.lower() in prediction_clean
                or prediction_clean in category.lower()
            ):
                return i

        # If no match found, return None
        print(f"    ✗ Could not map prediction '{prediction}' to any category")
        return None

    def _evaluate_distilbert(
        self, X_test, y_test, feature_name: str, model_type: str = "distilbert"
    ) -> Optional[Dict]:
        """Evaluate DistilBERT model"""
        model_path = f"models/{model_type}/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            return None

        # Get model configuration
        model_config = MODEL_CONFIGS_LOADED.get("distilbert", {})
        batch_size = model_config.get("params", {}).get("eval_batch_size", 64)

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
        dataloader = DataLoader(ds, batch_size=batch_size)

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

        # Get test_df for domain metrics if available
        test_df_for_metrics = None
        if self.dataset_name == "uci" and hasattr(self, "_current_test_df"):
            test_df_for_metrics = self._current_test_df

        metrics = self._calculate_metrics(
            y_encoded,
            predictions,
            len(texts),
            end_time - start_time,
            label_encoder,
            test_df_for_metrics,
        )

        metrics.update(
            {
                "model": model_type,
                "dataset": self.dataset_name,
                "feature": feature_name,
            }
        )

        return metrics

    def _evaluate_traditional(
        self, X_test, y_test, model_type: str, feature_name: str
    ) -> Optional[Dict]:
        """Evaluate traditional ML models"""
        # Special handling for distant labeling
        if model_type == "distant-labeling":
            return self._evaluate_distant_labeling(X_test, y_test, feature_name)

        model_path = f"models/{model_type}/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            return None

        # Load model and vectorizer
        model = torch.load(f"{model_path}/model.pt")
        label_encoder = torch.load(f"{model_path}/label_encoder.pt")
        vectorizer_type = torch.load(f"{model_path}/vectorizer_type.pt")

        # Get model configuration for any additional parameters
        model_config = MODEL_CONFIGS_LOADED.get(model_type, {})

        start_time = time.perf_counter()
        mask = None  # Initialize mask variable

        if vectorizer_type == "embeddings":
            embedder = self.load_embedder()
            # Filter long texts based on max length
            max_length = model_config.get("max_text_length", 2500)
            mask = X_test.str.len() < max_length
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

        # For UCI dataset with filtering, we need to align test_df with filtered data
        test_df_for_metrics = None
        if self.dataset_name == "uci" and hasattr(self, "_current_test_df"):
            if vectorizer_type == "embeddings" and mask is not None:
                # Filter test_df to match filtered data
                test_df_for_metrics = self._current_test_df[mask].reset_index(drop=True)
            else:
                test_df_for_metrics = self._current_test_df

        metrics = self._calculate_metrics(
            y_encoded,
            y_pred,
            len(y_test_final),
            end_time - start_time,
            label_encoder,
            test_df_for_metrics,
        )

        metrics.update(
            {"model": model_type, "dataset": self.dataset_name, "feature": feature_name}
        )

        return metrics

    def _evaluate_distant_labeling(
        self, X_test, y_test, feature_name: str
    ) -> Optional[Dict]:
        """Evaluate distant labeling model"""
        model_path = f"models/distant-labeling/{self.dataset_name}_{feature_name}"

        if not os.path.exists(model_path):
            print(f"Distant labeling model not found at {model_path}")
            return None

        # Load model components
        model = torch.load(f"{model_path}/model.pt")
        label_encoder = torch.load(f"{model_path}/label_encoder.pt")
        vectorizer = torch.load(f"{model_path}/vectorizer.pt")
        distant_classifier = torch.load(f"{model_path}/distant_classifier.pt")

        print(f"Evaluating distant labeling model on {len(X_test)} test samples...")

        start_time = time.perf_counter()

        # Transform features
        X_test_vec = vectorizer.transform(X_test)

        # Predict using the trained classifier
        y_pred = model.predict(X_test_vec)

        end_time = time.perf_counter()

        # Calculate metrics using the multi-class classification
        # We need to map the original test labels to the distant labeling categories

        # For evaluation, we'll use the original test labels as ground truth
        # and see how well the distant labeling model performs
        try:
            # Try to encode the test labels using the distant labeling label encoder
            y_test_encoded = label_encoder.transform(y_test)
        except ValueError:
            # If test labels don't match distant labeling categories,
            # we'll create a mapping or use a subset of test data
            print("Warning: Test labels don't match distant labeling categories")

            # Get the categories that the distant labeling model can predict
            distant_categories = set(label_encoder.classes_)

            # Filter test data to only include samples with categories we can predict
            valid_indices = []
            valid_y_test = []
            valid_X_test = []

            for i, label in enumerate(y_test):
                if label in distant_categories:
                    valid_indices.append(i)
                    valid_y_test.append(label)
                    valid_X_test.append(X_test.iloc[i])

            if len(valid_indices) == 0:
                print(
                    "No matching categories found between test set and distant labeling model"
                )
                return None

            # Re-run prediction on filtered data
            X_test_filtered = pd.Series(valid_X_test)
            X_test_vec_filtered = vectorizer.transform(X_test_filtered)
            y_pred_filtered = model.predict(X_test_vec_filtered)
            y_test_encoded = label_encoder.transform(valid_y_test)

            print(
                f"Evaluation limited to {len(valid_indices)} samples with matching categories"
            )

            # Use filtered data for metrics
            y_pred = y_pred_filtered
            y_test_final = y_test_encoded
        else:
            y_test_final = y_test_encoded

        # Calculate metrics
        metrics = self._calculate_metrics(
            y_test_final,
            y_pred,
            len(y_pred),
            end_time - start_time,
            label_encoder,
            None,
        )

        # Load full test dataset for additional metadata
        full_df, _ = load_data(self.dataset_name)
        train, val, test_df = make_splits(full_df, self.dataset_name)

        metrics.update(
            {
                "model": "distant-labeling",
                "dataset": self.dataset_name,
                "feature": feature_name,
                "distant_labeling_info": {
                    "uses_url_categorization": distant_classifier.uses_url_based_categorization(
                        test_df
                    ),
                    "n_url_labeled": len(distant_classifier.valid_categories),
                    "n_classified": len(y_pred),
                    "categories": list(label_encoder.classes_),
                },
            }
        )

        return metrics

    def _calculate_metrics(
        self, y_true, y_pred, n_samples, time_taken, label_encoder, test_df=None
    ) -> Dict:
        """Calculate evaluation metrics including per-topic and domain-level metrics"""
        # Get unique labels
        unique_labels = np.unique(y_true)
        label_names = label_encoder.inverse_transform(unique_labels)

        # Calculate per-topic metrics
        per_topic_metrics = {}
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        for i, _ in enumerate(unique_labels):
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

        result = {**macro_metrics, "per_topic": per_topic_metrics}

        # Add domain-level metrics for UCI dataset
        if (
            test_df is not None
            and self.dataset_name == "uci"
            and "PUBLISHER" in test_df.columns
        ):
            domain_metrics = self._calculate_domain_metrics(
                y_true, y_pred, test_df, label_encoder
            )
            result.update(domain_metrics)

        return result

    def _calculate_domain_metrics(self, y_true, y_pred, test_df, label_encoder) -> Dict:
        """Calculate domain-level and domain-and-topic-level performance metrics"""
        # Create DataFrame for easier manipulation
        results_df = test_df.copy()
        results_df["y_true_encoded"] = y_true
        results_df["y_pred_encoded"] = y_pred
        results_df["y_true"] = label_encoder.inverse_transform(y_true)
        results_df["y_pred"] = label_encoder.inverse_transform(y_pred)
        results_df["correct"] = y_true == y_pred

        # Calculate domain-level metrics (per publisher)
        domain_metrics = {}
        domain_topic_metrics = {}

        for publisher in results_df["PUBLISHER"].unique():
            publisher_data = results_df[results_df["PUBLISHER"] == publisher]

            if len(publisher_data) == 0:
                continue

            # Domain-level overall metrics
            pub_y_true = publisher_data["y_true_encoded"].values
            pub_y_pred = publisher_data["y_pred_encoded"].values

            domain_accuracy = accuracy_score(pub_y_true, pub_y_pred)
            domain_precision = precision_score(
                pub_y_true, pub_y_pred, average="macro", zero_division=0
            )
            domain_recall = recall_score(
                pub_y_true, pub_y_pred, average="macro", zero_division=0
            )
            domain_f1 = f1_score(
                pub_y_true, pub_y_pred, average="macro", zero_division=0
            )

            domain_metrics[publisher] = {
                "accuracy": domain_accuracy,
                "precision": domain_precision,
                "recall": domain_recall,
                "f1": domain_f1,
                "support": len(publisher_data),
                "n_correct": int(publisher_data["correct"].sum()),
            }

            # Domain-and-topic-level metrics (per publisher per topic)
            domain_topic_metrics[publisher] = {}

            for topic in publisher_data["y_true"].unique():
                topic_data = publisher_data[publisher_data["y_true"] == topic]

                if len(topic_data) == 0:
                    continue

                topic_y_true = topic_data["y_true_encoded"].values
                topic_y_pred = topic_data["y_pred_encoded"].values

                topic_accuracy = accuracy_score(topic_y_true, topic_y_pred)

                # For single-class metrics, we need to handle the case where there's only one class
                try:
                    topic_precision = precision_score(
                        topic_y_true, topic_y_pred, average="macro", zero_division=0
                    )
                    topic_recall = recall_score(
                        topic_y_true, topic_y_pred, average="macro", zero_division=0
                    )
                    topic_f1 = f1_score(
                        topic_y_true, topic_y_pred, average="macro", zero_division=0
                    )
                except Exception:
                    # Fallback for edge cases
                    topic_precision = topic_accuracy
                    topic_recall = topic_accuracy
                    topic_f1 = topic_accuracy

                domain_topic_metrics[publisher][topic] = {
                    "accuracy": topic_accuracy,
                    "precision": topic_precision,
                    "recall": topic_recall,
                    "f1": topic_f1,
                    "support": len(topic_data),
                    "n_correct": int(topic_data["correct"].sum()),
                }

        return {"per_domain": domain_metrics, "per_domain_topic": domain_topic_metrics}

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
    train, val, test = make_splits(df, dataset_name)

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both")
    parser.add_argument(
        "--datasets", nargs="+", default=["huffpo", "uci", "recognasumm"]
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument(
        "--features", nargs="+", default=list(FEATURE_EXTRACTORS_LOADED.keys())
    )
    parser.add_argument(
        "--run-id", 
        help="Unique identifier for this run (default: timestamp)",
        default=None
    )

    args = parser.parse_args()
    
    # Generate unique run identifier
    if args.run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = args.run_id

    # Print configuration summary
    print(f"\n{'=' * 50}")
    print("Configuration Summary")
    print(f"{'=' * 50}")
    print(f"Run ID: {run_id}")
    print(f"Models: {len(args.models)} - {', '.join(args.models)}")
    print(f"Features: {len(args.features)} - {', '.join(args.features)}")
    print(f"Datasets: {len(args.datasets)} - {', '.join(args.datasets)}")
    print(f"Mode: {args.mode}")
    print(f"Traditional: {[m for m in args.models if m in TRADITIONAL_MODELS]}")
    print(f"Deep Learning: {[m for m in args.models if m in DEEP_MODELS]}")
    print(f"LLM: {[m for m in args.models if m in LLM_MODELS]}")
    print(f"{'=' * 50}\n")

    results = []
    per_topic_results = []
    per_domain_results = []
    per_domain_topic_results = []

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
                    if model_type in LLM_MODELS:
                        print("  Setting up LLM (no training needed)...")
                        try:
                            trainer.train_llm_model(
                                X_train, train["y"], model_type, feature_name
                            )
                        except Exception as e:
                            print(f"  ✗ LLM setup failed: {str(e)}")
                            continue
                    else:
                        print("  Training...")
                        try:
                            if model_type in [
                                "distilbert",
                                "distilbert-1k",
                                "distilbert-3k",
                            ]:
                                X_val = trainer.prepare_features(val, feature_name)
                                trainer.train_distilbert(
                                    X_train,
                                    train["y"],
                                    X_val,
                                    val["y"],
                                    feature_name,
                                    model_type,
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

                            # Extract domain-level metrics
                            per_domain = metrics.pop("per_domain", {})
                            per_domain_topic = metrics.pop("per_domain_topic", {})

                            # Add run_id to all metrics for tracking
                            metrics["run_id"] = run_id
                            
                            # Add macro metrics to results
                            results.append(metrics)

                            # Add per-topic metrics to separate results
                            for topic, topic_metrics in per_topic.items():
                                per_topic_result = {
                                    "run_id": run_id,
                                    "dataset": metrics["dataset"],
                                    "model": metrics["model"],
                                    "feature": metrics["feature"],
                                    "topic": topic,
                                    **topic_metrics,
                                }
                                per_topic_results.append(per_topic_result)

                            # Add per-domain metrics to separate results
                            for domain, domain_metrics in per_domain.items():
                                per_domain_result = {
                                    "run_id": run_id,
                                    "dataset": metrics["dataset"],
                                    "model": metrics["model"],
                                    "feature": metrics["feature"],
                                    "domain": domain,
                                    **domain_metrics,
                                }
                                per_domain_results.append(per_domain_result)

                            # Add per-domain-topic metrics to separate results
                            for domain, domain_topics in per_domain_topic.items():
                                for topic, topic_metrics in domain_topics.items():
                                    per_domain_topic_result = {
                                        "run_id": run_id,
                                        "dataset": metrics["dataset"],
                                        "model": metrics["model"],
                                        "feature": metrics["feature"],
                                        "domain": domain,
                                        "topic": topic,
                                        **topic_metrics,
                                    }
                                    per_domain_topic_results.append(
                                        per_domain_topic_result
                                    )

                            success_info = ""
                            if "success_rate" in metrics:
                                success_info = (
                                    f", Success: {metrics['success_rate']:.1%}"
                                )

                            print(
                                f"  ✓ Evaluation completed - F1: {metrics['f1']:.3f}{success_info}"
                            )
                        else:
                            print("  ✗ Model not found for evaluation")
                    except Exception as e:
                        print(f"  ✗ Evaluation failed: {str(e)}")

    # Save results
    if results and args.mode in ["evaluate", "both"]:
        # Create output directory if it doesn't exist
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save macro results with unique filename
        results_df = pd.DataFrame(results)
        macro_filename = f"{output_dir}/unified_evaluation_results_{run_id}.csv"
        results_df.to_csv(macro_filename, index=False)
        print(f"\n{'=' * 50}")
        print(f"Macro results saved to {macro_filename}")

        # Save per-topic results
        if per_topic_results:
            per_topic_df = pd.DataFrame(per_topic_results)
            topic_filename = f"{output_dir}/per_topic_evaluation_results_{run_id}.csv"
            per_topic_df.to_csv(topic_filename, index=False)
            print(f"Per-topic results saved to {topic_filename}")
        else:
            print("No per-topic results to save.")
            per_topic_df = pd.DataFrame()

        # Save per-domain results
        if per_domain_results:
            per_domain_df = pd.DataFrame(per_domain_results)
            domain_filename = f"{output_dir}/per_domain_evaluation_results_{run_id}.csv"
            per_domain_df.to_csv(domain_filename, index=False)
            print(f"Per-domain results saved to {domain_filename}")
        else:
            print("No per-domain results to save.")

        # Save per-domain-topic results
        if per_domain_topic_results:
            per_domain_topic_df = pd.DataFrame(per_domain_topic_results)
            domain_topic_filename = f"{output_dir}/per_domain_topic_evaluation_results_{run_id}.csv"
            per_domain_topic_df.to_csv(domain_topic_filename, index=False)
            print(f"Per-domain-topic results saved to {domain_topic_filename}")
        else:
            print("No per-domain-topic results to save.")

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

        # Print domain-level summary for UCI dataset
        if per_domain_results and any(
            result["dataset"] == "uci" for result in per_domain_results
        ):
            print("\nTop 10 domain performers for UCI dataset:")
            uci_domains = [
                result for result in per_domain_results if result["dataset"] == "uci"
            ]
            if uci_domains:
                uci_domain_df = pd.DataFrame(uci_domains)
                top_domains = uci_domain_df.nlargest(10, "f1")[
                    ["domain", "model", "feature", "f1", "accuracy", "support"]
                ]
                print(top_domains.to_string(index=False))

        # Print domain-topic summary for UCI dataset
        if per_domain_topic_results and any(
            result["dataset"] == "uci" for result in per_domain_topic_results
        ):
            print("\nTop domain-topic combinations for UCI dataset:")
            uci_domain_topics = [
                result
                for result in per_domain_topic_results
                if result["dataset"] == "uci"
            ]
            if uci_domain_topics:
                uci_dt_df = pd.DataFrame(uci_domain_topics)
                top_domain_topics = uci_dt_df.nlargest(10, "f1")[
                    ["domain", "topic", "model", "feature", "f1", "accuracy", "support"]
                ]
                print(top_domain_topics.to_string(index=False))


if __name__ == "__main__":
    main()

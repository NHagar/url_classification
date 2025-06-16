"""
Configuration system for easily adding new models and features
"""

from typing import Any, Dict

import xgboost as xgb
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Model configurations with hyperparameters
MODEL_CONFIGS = {
    "distilbert": {
        "type": "deep",
        "params": {
            "epochs": 3,
            "batch_size": 16,
            "eval_batch_size": 64,
            "warmup_steps": 500,
            "weight_decay": 0.01,
        },
        "model_names": {
            "default": "distilbert-base-uncased",
            "recognasumm": "distilbert-base-multilingual-cased",
        },
    },
    "log-reg": {
        "type": "traditional",
        "vectorizer": "tfidf",
        "vectorizer_params": {"max_features": 5000},
        "model_class": "LogisticRegression",
        "model_params": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
    },
    "svm": {
        "type": "traditional",
        "vectorizer": "tfidf",
        "vectorizer_params": {
            "max_features": 2000,  # Reduced from 5000
            "min_df": 5,  # Ignore terms that appear in fewer than 5 documents
            "max_df": 0.95,  # Ignore terms that appear in more than 95% of documents
            "ngram_range": (1, 1),  # Only unigrams for speed
        },
        "model_class": "SVC",
        "model_params": {"kernel": "linear", "random_state": 42, "probability": False},  # Disabled probability for speed
    },
    "tree-ensemble": {
        "type": "traditional",
        "vectorizer": "tfidf",
        "vectorizer_params": {"max_features": 5000},
        "model_class": "RandomForestClassifier",
        "model_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    },
    "distant-labeling": {
        "type": "traditional",
        "vectorizer": "tfidf",
        "vectorizer_params": {"max_features": 10000, "ngram_range": (1, 2)},
        "model_class": "MultinomialNB",
        "model_params": {"alpha": 1.0},
    },
    "xgboost": {
        "type": "traditional",
        "vectorizer": "embeddings",
        "vectorizer_params": {},
        "embedding_model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "model_class": "XGBClassifier",
        "model_params": {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
        },
    },
    "gradient-boosting": {
        "type": "traditional",
        "vectorizer": "tfidf",
        "vectorizer_params": {"max_features": 5000},
        "model_class": "GradientBoostingClassifier",
        "model_params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
    },
    "llm-local": {
        "type": "llm",
        "params": {
            "model_name": "phi-4",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "temperature": 0,
            "batch_size": 10,  # Process in smaller batches for LLM
        },
        "prompt_template": {
            "system": "You are a text classifier. Given a text sample, classify it into one of the provided categories. Respond with only the category name, nothing else.",
            "user": "Categories: {categories}\n\nText to classify: {text}\n\nCategory:",
        },
    },
    "distilbert-1k": {
        "type": "deep",
        "subset_size": 1000,
        "random_seed": 42,
        "params": {
            "epochs": 3,
            "batch_size": 16,
            "eval_batch_size": 64,
            "warmup_steps": 500,
            "weight_decay": 0.01,
        },
        "model_names": {
            "default": "distilbert-base-uncased",
            "recognasumm": "distilbert-base-multilingual-cased",
        },
    },
    "distilbert-3k": {
        "type": "deep",
        "subset_size": 3000,
        "random_seed": 42,
        "params": {
            "epochs": 3,
            "batch_size": 16,
            "eval_batch_size": 64,
            "warmup_steps": 500,
            "weight_decay": 0.01,
        },
        "model_names": {
            "default": "distilbert-base-uncased",
            "recognasumm": "distilbert-base-multilingual-cased",
        },
    },
}

# Feature extraction configurations
FEATURE_EXTRACTORS = {
    "title_subtitle": {
        "description": "Concatenated title and subtitle",
        "extractors": {
            "huffpo": "lambda df: df['headline'] + ' ' + df['short_description']",
            "uci": "lambda df: df['TITLE']",
            "recognasumm": "lambda df: df['Titulo'] + ' ' + df['Subtitulo']",
        },
    },
    "title": {
        "description": "Title/headline only",
        "extractors": {
            "huffpo": "lambda df: df['headline']",
            "uci": "lambda df: df['TITLE']",
            "recognasumm": "lambda df: df['Titulo']",
        },
    },
    "snippet_description": {
        "description": "Short description or summary",
        "extractors": {
            "huffpo": "lambda df: df['short_description']",
            "uci": None,
            "recognasumm": "lambda df: df['Sumario']",
        },
    },
    "url_heading_subhead": {
        "description": "URL concatenated with heading and subheading",
        "extractors": {
            "huffpo": "lambda df: df['link'] + ' ' + df['headline'] + ' ' + df['short_description']",
            "uci": "lambda df: df['URL'] + ' ' + df['TITLE']",
            "recognasumm": "lambda df: df['URL'] + ' ' + df['Titulo'] + ' ' + df['Subtitulo']",
        },
    },
    "url_raw": {
        "description": "Raw URL as-is",
        "extractors": {
            "huffpo": "lambda df: df['link']",
            "uci": "lambda df: df['URL']",
            "recognasumm": "lambda df: df['URL']",
        },
    },
    "url_path_raw": {
        "description": "URL path component",
        "extractors": {
            "huffpo": "lambda df: df['x_path']",
            "uci": "lambda df: df['x_path']",
            "recognasumm": "lambda df: df['x_path']",
        },
    },
    "url_path_cleaned": {
        "description": "Cleaned URL path with special chars removed",
        "extractors": {
            "huffpo": "lambda df: df['x']",
            "uci": "lambda df: df['x']",
            "recognasumm": "lambda df: df['x']",
        },
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "huffpo": {
        "name": "HuffPost News",
        "language": "en",
        "min_category_percentage": 0.02,
        "test_size": 0.2,
        "val_size": 0.5,  # of test size
        "random_seed": 20240823,
    },
    "uci": {
        "name": "UCI News Dataset",
        "language": "en",
        "min_category_percentage": None,
        "test_size": 0.2,
        "val_size": 0.5,
        "random_seed": 20240823,
    },
    "recognasumm": {
        "name": "RecognaSumm Portuguese News",
        "language": "pt",
        "min_category_percentage": 0.02,
        "test_size": 0.2,
        "val_size": 0.5,
        "random_seed": 20240823,
    },
}


def save_configs(filename: str = "config/ml_configs.yaml"):
    """Save all configurations to a YAML file"""
    configs = {
        "models": MODEL_CONFIGS,
        "features": FEATURE_EXTRACTORS,
        "datasets": DATASET_CONFIGS,
    }

    with open(filename, "w") as f:
        yaml.dump(configs, f, default_flow_style=False, sort_keys=False)

    print(f"Configurations saved to {filename}")


def load_configs(filename: str = "config/ml_configs.yaml") -> Dict[str, Any]:
    """Load configurations from YAML file"""
    with open(filename, "r") as f:
        configs = yaml.safe_load(f)

    # Convert string lambdas back to functions
    for _, feature_config in configs["features"].items():
        for dataset, extractor_str in feature_config["extractors"].items():
            if (
                extractor_str
                and isinstance(extractor_str, str)
                and extractor_str.startswith("lambda")
            ):
                # Safe evaluation of lambda functions
                feature_config["extractors"][dataset] = eval(extractor_str)

    return configs


def get_model_instance(config: Dict[str, Any]):
    """Create a model instance based on configuration"""

    model_classes = {
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "MultinomialNB": MultinomialNB,
        "XGBClassifier": xgb.XGBClassifier,
    }

    model_class_name = config.get("model_class")
    model_params = config.get("model_params", {})

    if model_class_name in model_classes:
        return model_classes[model_class_name](**model_params)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


def add_new_model(name: str, config: Dict[str, Any]):
    """Add a new model configuration"""
    MODEL_CONFIGS[name] = config
    print(f"Added new model: {name}")


def add_new_feature(name: str, description: str, extractors: Dict[str, str]):
    """Add a new feature configuration"""
    FEATURE_EXTRACTORS[name] = {"description": description, "extractors": extractors}
    print(f"Added new feature: {name}")


# Example usage
if __name__ == "__main__":
    # Save current configurations
    save_configs()

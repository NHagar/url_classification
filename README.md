# URL Classification Project

A machine learning project for classifying news articles based on URLs, titles, and content features. This repository contains tools for training various classification models, analyzing dataset statistics, and understanding URL structure patterns.

## Repository Structure

```
url_classification/
    README.md                           # This file
    pyproject.toml                      # Python project configuration and dependencies
    uv.lock                            # Dependency lock file
    config/
        ml_configs.yaml                # Machine learning model configurations
    data/
        raw/                           # Original datasets
            news_categories.parquet    # HuffPost news categories
            recognasumm.parquet        # RecognaSumm dataset
            uci_categories.parquet     # UCI news aggregator dataset
        processed/                     # Generated outputs and results
            *_stats.txt                # Dataset statistics reports
            evaluation_metrics*.csv    # Model evaluation results
            url_structure_analysis.png # URL structure visualizations
            *_test.csv                 # Test set predictions
    models/                            # Trained model artifacts
    results/                           # Analysis outputs and visualizations
    url_classification/                # Main Python package
        __init__.py
        dataset_loading.py             # Data loading utilities
        distant_labeling.py            # Distant supervision algorithms  
        model_config.py                # Model configuration management
        train_and_evaluate.py          # Core training and evaluation script
        viz.py                         # Visualization utilities
    adhoc/                             # Experimental notebooks and scripts
    tests/                             # Test files
    dataset_descriptive_stats.py       # Dataset analysis script
    url_structure_analysis.py          # URL pattern analysis script
    *.sh                               # Job submission scripts
```

## Main Scripts

### 1. Training and Evaluation Script

**Location**: `url_classification/train_and_evaluate.py`

**Purpose**: Train and evaluate various machine learning models for URL/text classification.


### 2. Dataset Descriptive Statistics Script

**Location**: `dataset_descriptive_stats.py`

**Purpose**: Generate comprehensive statistics about the datasets including domain distribution, category balance, and cross-dataset comparisons.

### 3. URL Structure Analysis Script

**Location**: `url_structure_analysis.py`

**Purpose**: Analyze URL structure patterns and their correlation with model performance.

## Data Requirements

The project expects datasets in the `data/raw/` directory:
- `news_categories.parquet` - HuffPost news categories
- `recognasumm.parquet` - RecognaSumm dataset  
- `uci_categories.parquet` - UCI news aggregator dataset

## Model Configuration

Model settings are managed through:
- `config/ml_configs.yaml` - YAML configuration file
- `url_classification/model_config.py` - Python configuration management

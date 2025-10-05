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

## Computational Requirements and Runtime

### Resource Requirements by Model Type

The computational demands vary significantly across models. Below are the resource allocations used in the original study (see SLURM scripts: [gpu.sh](gpu.sh), [short.sh](short.sh), [long.sh](long.sh)):

**Deep Learning Models (distilbert, distilbert-1k, distilbert-3k, xgboost)**
- **Hardware**: GPU (A100) with 8 CPUs
- **Memory**: ~16GB RAM
- **Time**: Up to 8 hours for full runs
- **Storage**: ~10-15GB free space required
- **Note**: These models are **impractical to run on standard laptops** due to computational demands

**Traditional ML Models (log-reg, svm, tree-ensemble, distant-labeling, gradient-boosting)**
- **Hardware**: CPU-only (8-16 cores)
- **Memory**: ~16GB RAM
- **Time**: 1-4 hours for full runs
- **Storage**: ~5-10GB free space required
- **Note**: Can run on standard laptops but may take significantly longer

### Recommended Reproduction Scenarios

For easier reproduction, consider these scenarios:

**Scenario 1: Quick Test (Laptop-friendly)**
```bash
# Run lightweight models only (~45-90 minutes on standard laptop)
uv run -m url_classification.train_and_evaluate \
  --models log-reg svm distant-labeling \
  --datasets huffpo
```
**Requirements**: 16GB RAM, SSD with 5GB free space

**Scenario 2: Traditional Models Only (No GPU)**
```bash
# All non-deep learning models (~2-4 hours on standard laptop)
uv run -m url_classification.train_and_evaluate \
  --models log-reg svm tree-ensemble distant-labeling gradient-boosting \
  --mode evaluate
```
**Requirements**: 16GB RAM, 8+ CPU cores, SSD with 10GB free space

**Scenario 3: Full Reproduction (Requires HPC/GPU)**
```bash
# Complete replication including deep learning models
uv run -m url_classification.train_and_evaluate \
  --models distilbert distilbert-1k distilbert-3k xgboost \
  --mode evaluate
```
**Requirements**: GPU (preferably A100 or similar), 16GB+ RAM, 15GB+ storage

### Configuring Models and Features

To include/exclude specific models or features:

**Via Command Line:**
```bash
# Specify models
uv run -m url_classification.train_and_evaluate --models log-reg svm

# Specify features
uv run -m url_classification.train_and_evaluate --features url_raw title

# Specify datasets
uv run -m url_classification.train_and_evaluate --datasets huffpo uci
```

**Via Configuration Files:**
- Comment out models in `config/ml_configs.yaml` to exclude them
- Comment out models in `url_classification/model_config.py` to exclude them

**Note**: By default, distilbert and xgboost are included but may be excluded for laptop-based reproduction.

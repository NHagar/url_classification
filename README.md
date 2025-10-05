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

## Main Scripts and Paper Artifacts

This section maps each table and figure in the paper to the script(s) used to generate it.

### Table 1: Descriptive Statistics

**Script**: `dataset_descriptive_stats.py`

**Output**: Descriptive statistics for the filtered subset of the three benchmarking datasets (rows, unique domains, topics, Gini coefficient, topic label entropy)

**Run**:
```bash
uv run python dataset_descriptive_stats.py
```

### Table 2: F1 Scores for Model-Feature Combinations

**Script**: `url_classification/train_and_evaluate.py`

**Output**: F1 scores for all combinations of models and input features across datasets, including URL-only classification results

**Run**:
```bash
# Run all models and features (full replication)
uv run -m url_classification.train_and_evaluate --mode evaluate

# Or run specific models
uv run -m url_classification.train_and_evaluate \
  --models distilbert log-reg svm tree-ensemble gradient-boosting xgboost \
  --mode evaluate
```

**Output files**: `data/processed/evaluation_metrics*.csv`

### Table 3: F1 Scores by Topic and Feature

**Script**: `url_classification/train_and_evaluate.py` (same as Table 2)

**Output**: F1 scores for all combinations of topics and input features across datasets

**Note**: Per-topic results are included in the same evaluation output as Table 2

### Table 4: Impact of Date Removal

**Script**: `date_ablation_study.py`

**Output**: Logistic regression classifier performance with and without dates in URL paths

**Run**:
```bash
# Run with default settings (distilbert on uci dataset with url_path_raw)
uv run python date_ablation_study.py

# Or specify datasets, models, and features
uv run python date_ablation_study.py \
  --datasets huffpo uci recognasumm \
  --models log-reg \
  --features url_path_raw \
  --mode both
```

**Output files**:
- `data/processed/date_ablation_results_*.csv` (full results)
- `data/processed/date_ablation_comparison_*.csv` (summary comparison)

### Figure 1: Model Throughput

**Script**: `url_classification/train_and_evaluate.py`

**Output**: Prediction throughput (predictions per second) for each model across datasets

**Note**: Throughput metrics are captured during model evaluation

### Figure 2: Training Data Ablation

**Script**: `url_classification/train_and_evaluate.py` with subset models

**Output**: Effect of training data size on DistilBERT classifier performance

**Run**:
```bash
uv run -m url_classification.train_and_evaluate \
  --models distilbert distilbert-1k distilbert-3k \
  --mode evaluate
```

### Additional Analysis Scripts

**URL Structure Analysis**: `url_structure_analysis.py`
- Analyzes URL structure patterns and their correlation with model performance
- Generates visualizations of URL components

**Run**:
```bash
uv run python url_structure_analysis.py
```

## Setup and Installation

### Prerequisites

**Python Version**: This project requires **Python 3.12**. While the `pyproject.toml` specifies `>=3.10`, the locked dependencies (particularly PyTorch and related packages) are known to have compatibility issues with Python 3.13+.

**Recommended Setup:**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 using uv
uv python install 3.12

# Create environment and install dependencies with Python 3.12
uv sync --python 3.12
```

### Common Setup Issues

**PyTorch Compatibility Issue**
If you encounter PyTorch-related errors after initial setup, you may need to reinstall PyTorch:
```bash
uv pip install --force-reinstall torch==2.4.0
```

**Missing Directories**
The scripts expect a `data/processed/` directory to exist. Create it if needed:
```bash
mkdir -p data/processed
```

**Verification**
After setup, verify your environment:
```bash
python --version  # Should show Python 3.12.x
uv run python -c "import torch; print(torch.__version__)"  # Should show 2.4.0
```

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

## Package Dependencies

The project uses the following key packages (complete list in `pyproject.toml`):

### Core Dependencies
```
python = 3.12
torch = 2.4.0
transformers = 4.44.0
scikit-learn = 1.5.1
xgboost = 2.1.1
sentence-transformers = 3.0.1
```

### Data Processing
```
pandas = 2.2.2
numpy = 1.26.4
pyarrow = 17.0.0
duckdb = 1.0.0
datasets = 2.21.0
```

### Visualization
```
matplotlib >= 3.10.3
seaborn >= 0.13.2
```

### Other Key Packages
```
accelerate = 0.33.0
huggingface-hub = 0.24.5
evaluate = 0.4.2
tokenizers = 0.19.1
safetensors = 0.4.4
tldextract >= 5.3.0
pyyaml = 6.0.2
```

For a complete list of all dependencies with exact versions, see `pyproject.toml` or run:
```bash
uv pip list
```

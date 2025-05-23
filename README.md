# Unified Model Training and Evaluation System

## Overview

This system provides a streamlined approach to training and evaluating multiple machine learning models across various text features and datasets. It replaces the previous fragmented approach with a unified, modular architecture.

## Architecture

### Core Components

1. **UnifiedModelTrainer**: Handles training for all model types
   - Supports both traditional ML and deep learning models
   - Automatic feature extraction and preprocessing
   - Smart handling of missing features
   - Device-aware training (GPU/MPS/CPU)

2. **UnifiedModelEvaluator**: Standardized evaluation across all models
   - Consistent metrics calculation
   - Throughput measurement
   - Batch processing for efficiency

3. **Feature Configuration System**: Flexible feature mapping
   - Dataset-specific feature extractors
   - Graceful handling of unavailable features
   - Easy to extend with new features

### Key Design Decisions

#### 1. Feature Mapping Architecture
```python
TEXT_FEATURES_CONFIG = {
    "feature_name": {
        "dataset1": lambda df: extraction_logic,
        "dataset2": lambda df: extraction_logic,
        "dataset3": lambda df: None  # Not available
    }
}
```

This design allows:
- Easy addition of new features
- Clear visibility of feature availability
- Dataset-specific extraction logic
- Automatic skipping of unavailable features

#### 2. Model Abstraction
All models follow a consistent interface:
- Traditional models: TF-IDF vectorization (except XGBoost with embeddings)
- Deep models: Tokenization with appropriate pre-trained models
- Consistent save/load patterns

#### 3. Smart Vectorization Strategy
- **Distant Labeling**: TF-IDF with 10k features
- **Log-Reg/SVM/Tree**: TF-IDF with 5k features  
- **XGBoost**: Embeddings for text features, TF-IDF for URL features
- **DistilBERT**: Native tokenization

## Supported Models

| Model | Type | Vectorization | Best For |
|-------|------|---------------|----------|
| DistilBERT | Deep Learning | Tokenization | High accuracy, context understanding |
| Logistic Regression | Traditional | TF-IDF | Fast, interpretable, baseline |
| SVM | Traditional | TF-IDF | Non-linear patterns, robust |
| Tree Ensemble | Traditional | TF-IDF | Feature interactions, robust |
| Distant Labeling | Traditional | TF-IDF | Weakly supervised scenarios |
| XGBoost | Gradient Boosting | Embeddings/TF-IDF | High performance, efficiency |

## Text Features

| Feature | Description | Example |
|---------|-------------|---------|
| title_subtitle | Combined title and subtitle | "Breaking News: Major Event Unfolds" |
| title | Title only | "Breaking News" |
| snippet_description | Short description/summary | "A major event occurred today..." |
| url_heading_subhead | URL + title + subtitle | "example.com/news Breaking News: Major Event" |
| url_raw | Complete URL | "https://example.com/news/story" |
| url_path_raw | URL path component | "/news/story" |
| url_path_cleaned | Cleaned path | "news story" |

## Dataset Support

### HuffPost (huffpo)
- English news articles
- All features available
- Categories: Politics, Entertainment, etc.

### UCI News (uci)
- English news dataset
- Limited features (no snippet/description)
- Academic dataset

### RecognaSumm (recognasumm)
- Portuguese news articles
- All features available
- Uses multilingual DistilBERT

## Usage Patterns

### Basic Training
```bash
# Train all models on all features
python -m url_classification.model_training.unified_system --mode train

# Train specific combinations
python -m url_classification.model_training.unified_system \
    --mode train \
    --datasets huffpo uci \
    --models distilbert xgboost \
    --features title url_raw
```

### Evaluation
```bash
# Evaluate all trained models
python -m url_classification.model_training.unified_system --mode evaluate

# Generate visualizations
python -m url_classification.analysis.unified_viz
```

## Performance Considerations

1. **Batch Processing**: All models use appropriate batch sizes
2. **Device Selection**: Automatic GPU/MPS detection
3. **Memory Management**: Text length filtering for embedding models
4. **Progress Tracking**: TQDM progress bars for long operations

## Extending the System

### Adding a New Model
1. Add model name to `ALL_MODELS` list
2. Implement training logic in `train_traditional_model()` or create new method
3. Implement evaluation logic in `_evaluate_traditional()` or create new method

### Adding a New Feature
1. Add feature to `TEXT_FEATURES_CONFIG`
2. Define extraction logic for each dataset
3. Set to `None` for unavailable features

### Adding a New Dataset
1. Update `load_dataset_with_features()`
2. Add feature mappings in `TEXT_FEATURES_CONFIG`
3. Handle any special preprocessing needs

## Output Structure

```
models/
├── distilbert/
│   ├── huffpo_title/
│   ├── huffpo_url_raw/
│   └── ...
├── xgboost/
│   ├── uci_title_subtitle/
│   └── ...
└── log-reg/
    └── ...

data/processed/
├── unified_evaluation_results.csv
├── huffpo_test.csv
├── f1_heatmaps_by_dataset.png
├── model_average_performance.png
└── ...
```

## Metrics Tracked

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction quality (macro average)
- **Recall**: Coverage of positive cases (macro average)
- **F1 Score**: Harmonic mean of precision and recall (macro average)
- **Throughput**: Samples processed per second

## Visualization Suite

1. **F1 Heatmaps**: Model vs Feature performance by dataset
2. **Model Comparison**: Average metrics across all experiments
3. **Feature Importance**: Which features work best overall
4. **Throughput Analysis**: Speed comparison (log scale)
5. **Top Combinations**: Best performing model-feature pairs
6. **Performance Distributions**: Statistical view of results
7. **Feature Availability Matrix**: What's available where

## Best Practices

1. **Start Small**: Test with one dataset and a few models first
2. **Monitor Progress**: Use progress bars to track long operations
3. **Check Availability**: Verify features exist before training
4. **Save Incrementally**: Models are saved after each training
5. **Analyze Results**: Use visualization suite for insights
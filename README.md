# Unified Model Training and Evaluation System

## Overview

This system provides a streamlined approach to training and evaluating multiple machine learning models across various text features and datasets. It replaces the previous fragmented approach with a unified, modular architecture. Now includes support for local LLM evaluation via LMStudio.

## Architecture

### Core Components

1. **UnifiedModelTrainer**: Handles training for all model types
   - Supports both traditional ML and deep learning models
   - Automatic feature extraction and preprocessing
   - Smart handling of missing features
   - Device-aware training (GPU/MPS/CPU)
   - LLM setup (no training needed, only label encoding)

2. **UnifiedModelEvaluator**: Standardized evaluation across all models
   - Consistent metrics calculation
   - Throughput measurement
   - Batch processing for efficiency
   - LLM evaluation via OpenAI API to LMStudio

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
- LLM models: Direct API calls with prompt engineering
- Consistent save/load patterns

#### 3. Smart Vectorization Strategy
- **Distant Labeling**: TF-IDF with 10k features
- **Log-Reg/SVM/Tree**: TF-IDF with 5k features  
- **XGBoost**: Embeddings for text features, TF-IDF for URL features
- **DistilBERT**: Native tokenization
- **LLM**: Direct text processing with prompt templates

## Supported Models

| Model | Type | Vectorization | Best For |
|-------|------|---------------|----------|
| DistilBERT | Deep Learning | Tokenization | High accuracy, context understanding |
| Logistic Regression | Traditional | TF-IDF | Fast, interpretable, baseline |
| SVM | Traditional | TF-IDF | Non-linear patterns, robust |
| Tree Ensemble | Traditional | TF-IDF | Feature interactions, robust |
| Distant Labeling | Traditional | TF-IDF | Weakly supervised scenarios |
| XGBoost | Gradient Boosting | Embeddings/TF-IDF | High performance, efficiency |
| LLM-Local | Large Language Model | Prompt-based | Zero-shot classification, interpretable |

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

## LLM Integration

### Setup Requirements

1. **Install LMStudio** from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download phi-4 model** (or your preferred model) in LMStudio
3. **Start the local server** on port 1234 (default)
4. **Install dependencies**: `pip install openai>=1.0.0`

### Configuration

The LLM is configured in `config/ml_configs.yaml`:

```yaml
llm-local:
  type: llm
  params:
    model_name: phi-4
    base_url: http://localhost:1234/v1
    api_key: lm-studio
    temperature: 0.1
    max_tokens: 50
    timeout: 30
    batch_size: 10
    max_text_length: 4000
  prompt_template:
    system: "You are a text classifier..."
    user: "Categories: {categories}\n\nText: {text}\n\nCategory:"
```

### LLM Features

- **Zero-shot classification**: No training required
- **Configurable prompts**: Customize system and user prompts
- **Batch processing**: Efficient handling of multiple samples
- **Error handling**: Robust API error management
- **Success rate tracking**: Monitor prediction quality
- **Text length filtering**: Handle long texts appropriately

## Usage Patterns

### Basic Training and Evaluation
```bash
# Train all models on all features (LLM will be set up, not trained)
python -m url_classification.train_and_evaluate --mode train

# Evaluate all models including LLM
python -m url_classification.train_and_evaluate --mode evaluate

# Both training and evaluation
python -m url_classification.train_and_evaluate --mode both
```

### Specific Model Combinations
```bash
# Train/evaluate specific combinations
python -m url_classification.train_and_evaluate \
    --mode both \
    --datasets huffpo uci \
    --models distilbert xgboost llm-local \
    --features title url_raw

# Only evaluate LLM (no training needed)
python -m url_classification.train_and_evaluate \
    --mode evaluate \
    --models llm-local \
    --features title_subtitle
```

### LLM-Only Evaluation
```bash
# Quick LLM evaluation on specific dataset
python -m url_classification.train_and_evaluate \
    --mode evaluate \
    --datasets huffpo \
    --models llm-local \
    --features title
```

### Generate Visualizations
```bash
python -m url_classification.viz
```

## Performance Considerations

1. **Batch Processing**: All models use appropriate batch sizes
2. **Device Selection**: Automatic GPU/MPS detection for deep models
3. **Memory Management**: Text length filtering for embedding and LLM models
4. **Progress Tracking**: TQDM progress bars for long operations
5. **LLM Optimization**: Configurable batch sizes and timeouts

## Extending the System

### Adding a New Model

#### Traditional/Deep Learning Model
1. Add model name to `MODEL_CONFIGS` in `model_config.py`
2. Implement training logic in `train_traditional_model()` or create new method
3. Implement evaluation logic in `_evaluate_traditional()` or create new method

#### LLM Model
1. Add model configuration to `MODEL_CONFIGS` with type "llm"
2. Configure API parameters and prompt templates
3. No training implementation needed
4. Evaluation handled automatically by `_evaluate_llm()`

### Adding a New Feature
1. Add feature to `FEATURE_EXTRACTORS` in `model_config.py`
2. Define extraction logic for each dataset
3. Set to `None` for unavailable features

### Adding a New Dataset
1. Update `load_dataset_with_features()` in `dataset_loading.py`
2. Add feature mappings in `FEATURE_EXTRACTORS`
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
├── llm-local/
│   ├── huffpo_title/
│   │   ├── label_encoder.pt
│   │   └── categories.pt
│   └── ...
└── log-reg/
    └── ...

data/processed/
├── unified_evaluation_results.csv
├── per_topic_evaluation_results.csv
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
- **Success Rate**: For LLM models, percentage of valid predictions

## Visualization Suite

1. **F1 Heatmaps**: Model vs Feature performance by dataset
2. **Model Comparison**: Average metrics across all experiments
3. **Feature Importance**: Which features work best overall
4. **Throughput Analysis**: Speed comparison (log scale)
5. **Top Combinations**: Best performing model-feature pairs
6. **Performance Distributions**: Statistical view of results
7. **Feature Availability Matrix**: What's available where

## Troubleshooting

### LLM Issues

**Connection Errors**:
- Ensure LMStudio server is running on `http://localhost:1234`
- Check if the correct model is loaded
- Verify API key configuration

**Low Success Rate**:
- Adjust prompt templates for better classification
- Increase `max_tokens` if responses are truncated
- Lower `temperature` for more consistent responses

**Timeout Errors**:
- Increase `timeout` parameter
- Reduce `batch_size` for faster processing
- Check system resources

**Category Mapping Issues**:
- Ensure categories in prompt match dataset labels exactly
- Check for case sensitivity issues
- Review LLM responses for unexpected formats

## Testing LLM Integration

Before running full evaluations, test your LLM setup:

```bash
# Test connection to LMStudio
python test_llm.py --connection-only

# Run full classification test
python test_llm.py

# Test with custom model
python test_llm.py --model-name your-model-name
```

## Best Practices

1. **Start Small**: Test with one dataset and a few models first
2. **Monitor Progress**: Use progress bars to track long operations
3. **Check Availability**: Verify features exist before training
4. **Save Incrementally**: Models are saved after each training
5. **Analyze Results**: Use visualization suite for insights
6. **LLM Optimization**: Start with conservative settings and adjust based on results
7. **Resource Management**: Monitor API costs and local resource usage for LLM
8. **Test LLM First**: Use the test script to verify LLM setup before full runs
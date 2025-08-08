# Classification Tools for Vision-Language Models

This directory contains tools for evaluating and analyzing the classification capabilities of Vision-Language Models (VLMs) on image datasets, with special attention to detecting and analyzing model hallucinations.

## Overview

The classification pipeline consists of three main components:

1. **Model Evaluation** (`evaluate_labels_only.py`) - Run VLM inference on images to predict labels
2. **Interactive Visualization** (`label_viewer.py`) - Explore results using FiftyOne's web interface
3. **Analysis Dashboard** (`label_results_analysis.py`) - Generate comprehensive analysis reports with visualizations

## Key Features

- **Hallucination Detection**: Automatically identifies when models predict labels that don't exist in the ground truth dataset (support = 0)
- **Performance Metrics**: Calculates precision, recall, F1 score, and accuracy at both sample and dataset levels
- **Interactive Exploration**: Browse results with powerful filtering and visualization capabilities
- **Comparative Analysis**: Compare multiple models side-by-side with detailed statistics and visualizations

## Quick Start

### 1. Evaluate a Model

Run VLM inference on your dataset:

```bash
# Evaluate on COCO validation set (default)
uv run python evaluate_labels_only.py --model "google/gemma-3-12b-it" --output results.json

# Evaluate on custom FiftyOne dataset
uv run python evaluate_labels_only.py --model "google/gemma-3-12b-it" --dataset-name my_dataset --output results.json

# Limit to specific number of samples for testing
uv run python evaluate_labels_only.py --model "google/gemma-3-12b-it" --max-samples 100 --output results.json
```

### 2. Visualize Results

Launch the interactive FiftyOne viewer:

```bash
# View results from COCO evaluation
uv run python label_viewer.py --results results.json

# View results from custom dataset
uv run python label_viewer.py --results results.json --source-dataset my_dataset

# Open specific view (e.g., samples with hallucinations)
uv run python label_viewer.py --results results.json --view with_hallucinations
```

Available views:
- `all` - All samples
- `perfect_matches` - Samples where predictions exactly match ground truth
- `high_f1` - High performance samples (F1 > 0.8)
- `low_f1` - Low performance samples (F1 < 0.3)
- `with_false_positives` - Samples containing false positives
- `with_false_negatives` - Samples containing false negatives
- `with_hallucinations` - Samples containing hallucinated labels
- `slowest_processing` - Samples sorted by processing time

### 3. Generate Analysis Dashboard

Create comprehensive analysis reports:

```bash
# Analyze single model results
uv run python label_results_analysis.py results.json

# Compare multiple models
uv run python label_results_analysis.py model1_results.json model2_results.json model3_results.json

# Specify custom output directory
uv run python label_results_analysis.py results.json --output-dir my_analysis
```

## Understanding Hallucinations

A key feature of these tools is the detection and analysis of **hallucinated labels** - categories that the model predicts but which never appear in the ground truth dataset. These are identified by having `support = 0` in the per-category metrics.

### Why Hallucinations Matter

- **Model Reliability**: High hallucination rates indicate the model may be unreliable
- **Training Issues**: Can reveal problems with training data or model architecture
- **Deployment Risks**: Hallucinations in production can lead to incorrect decisions

### Hallucination Metrics

The tools provide several hallucination-specific metrics:

- **Hallucinated Categories**: Total number of categories predicted but not in ground truth
- **Hallucination Rate**: Percentage of samples containing at least one hallucination
- **Average Hallucinations per Sample**: Mean number of hallucinated labels per image
- **Hallucination Distribution**: Histogram showing frequency of hallucinations

## Output Formats

### Evaluation Results (JSON)

The evaluation script produces a comprehensive JSON file containing:

```json
{
  "evaluation_config": {
    "model_path": "model_name",
    "target_categories": ["cat1", "cat2", ...],
    "evaluation_date": "timestamp"
  },
  "dataset_metrics": {
    "overall": {
      "precision": 0.85,
      "recall": 0.73,
      "f1_score": 0.78,
      "accuracy": 0.81
    },
    "per_category": {
      "category_name": {
        "precision": 0.9,
        "recall": 0.8,
        "f1": 0.85,
        "support": 100  // 0 indicates hallucinated category
      }
    }
  },
  "sample_results": [
    {
      "image_path": "path/to/image.jpg",
      "ground_truth_labels": ["cat", "dog"],
      "predicted_labels": ["cat", "dog", "bird"],
      "metrics": {...},
      "processing_time": 0.234
    }
  ]
}
```

### FiftyOne Dataset Fields

When viewing results in FiftyOne, each sample includes:

- `predicted_labels` - List of labels predicted by the model
- `ground_truth_labels` - List of actual labels
- `hallucinated_labels` - Labels predicted but not in ground truth dataset
- `has_hallucination` - Boolean flag for easy filtering
- `num_hallucinated` - Count of hallucinated labels
- `precision`, `recall`, `f1_score` - Per-sample metrics
- `perfect_match` - Whether predictions exactly match ground truth

### Analysis Dashboard

The HTML dashboard includes:

- **Overall Performance Metrics** - Precision, recall, F1, accuracy
- **Hallucination Statistics** - Comprehensive hallucination analysis
- **Category Performance** - Top/bottom performing categories with F1 scores
- **Error Analysis** - Most common false positives and false negatives
- **Visualizations** - Distribution plots, performance charts, hallucination analysis
- **Model Comparison** - Side-by-side comparison when analyzing multiple models

## Advanced Usage

### Resume Interrupted Evaluation

If evaluation is interrupted, resume from checkpoint:

```bash
uv run python evaluate_labels_only.py --model "model_name" --resume-from results_checkpoint.json --output results.json
```

### Custom Temperature Settings

Adjust model temperature for more/less creative responses:

```bash
uv run python evaluate_labels_only.py --model "model_name" --temperature 0.7 --output results.json
```

### Batch Processing

For large datasets, adjust batch size:

```bash
uv run python evaluate_labels_only.py --model "model_name" --batch-size 50 --output results.json
```

## Tips for Best Results

1. **Start Small**: Test with `--max-samples 10` to verify everything works
2. **Monitor Progress**: The evaluation script shows a progress bar and saves checkpoints
3. **Check Hallucinations**: Always review hallucinated categories to understand model behavior
4. **Compare Models**: Use the analysis dashboard to identify the best model for your use case
5. **Filter Results**: Use FiftyOne's powerful filtering to focus on problematic samples

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use a smaller model
2. **Slow Processing**: Some models are slower; check processing times in results
3. **Missing Categories**: Ensure your dataset has ground truth labels properly formatted
4. **FiftyOne Port Conflicts**: Use `--port 5152` to specify a different port

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
uv run python evaluate_labels_only.py --model "model_name" --verbose --output results.json
uv run python label_viewer.py --results results.json --verbose
```
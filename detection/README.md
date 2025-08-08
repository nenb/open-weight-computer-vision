# Object Detection Tools for Vision-Language Models

This directory contains tools for evaluating and analyzing the object detection capabilities of Vision-Language Models (VLMs) on image datasets, with special attention to bounding box localization.

## Overview

The object detection pipeline consists of four main components:

1. **Model Evaluation** (`detect_locations.py`) - Run VLM inference on images to detect objects with bounding boxes
2. **Interactive Visualization** (`detection_viewer.py`) - Explore results using FiftyOne's web interface
3. **Static Dashboard** (`bbox_results_analysis.py`) - Generate static HTML reports with visualizations

## Key Features

- **Evaluation**: Metrics including AP at multiple IoU thresholds
- **Bounding Box Visualization**: See predicted and ground truth boxes overlaid on images
- **Performance Analysis**: Detailed metrics by object category, size, and IoU threshold
- **Comparative Analysis**: Compare multiple models with comprehensive visualizations

## Quick Start

### 1. Evaluate a Model

Run VLM object detection on your dataset:

```bash
# Evaluate on COCO validation set using label evaluation results
uv run python detect_locations.py --model "google/paligemma3-3b-ft-cococap-896" \
    --evaluation-results ../classification/label_results.json \
    --output detection_results.json

# Limit to specific number of samples for testing
uv run python detect_locations.py --model "google/paligemma3-3b-ft-cococap-896" \
    --evaluation-results ../classification/label_results.json \
    --max-samples 100 \
    --output detection_results.json

# Resume from checkpoint if interrupted
uv run python detect_locations.py --model "google/paligemma3-3b-ft-cococap-896" \
    --evaluation-results ../classification/label_results.json \
    --resume-from detection_results_checkpoint.json \
    --output detection_results.json
```

### 2. Visualize Results

Launch the interactive FiftyOne viewer:

```bash
# View detection results
uv run python detection_viewer.py --results detection_results.json

# Open specific view
uv run python detection_viewer.py --results detection_results.json --view high_confidence

# Use custom port if default is occupied
uv run python detection_viewer.py --results detection_results.json --port 5152
```

Available views:
- `all` - All samples
- `with_detections` - Samples with at least one detection
- `no_detections` - Samples with no detections
- `high_confidence` - Detections with confidence > 0.8
- `low_confidence` - Detections with confidence < 0.3
- `many_objects` - Samples with many predicted objects (>5)
- `few_objects` - Samples with few predicted objects (1-2)
- `slowest_processing` - Samples sorted by processing time

### 3. Generate Static Analysis Report

Create HTML reports for sharing:

```bash
# Analyze single model results
uv run python bbox_results_analysis.py detection_results.json

# Compare multiple models
uv run python bbox_results_analysis.py model1_results.json model2_results.json model3_results.json

# Specify custom output directory
uv run python bbox_results_analysis.py detection_results.json --output-dir my_analysis
```

## Understanding Object Detection Metrics

### COCO Metrics

The tools use standard COCO evaluation metrics:

- **AP (Average Precision)**: Primary metric, averaged over IoU thresholds 0.5-0.95
- **AP50**: Average Precision at IoU threshold 0.5 (less strict)
- **AP75**: Average Precision at IoU threshold 0.75 (more strict)
- **APs/APm/APl**: AP for small/medium/large objects

### IoU (Intersection over Union)

IoU measures how well predicted bounding boxes overlap with ground truth:
- IoU = 1.0: Perfect overlap
- IoU > 0.5: Generally considered a correct detection
- IoU < 0.5: Poor localization

## Output Formats

### Detection Results (JSON)

The evaluation script produces a comprehensive JSON file:

```json
{
  "model_config": {
    "model_path": "model_name",
    "evaluation_date": "timestamp"
  },
  "coco_evaluation": {
    "metrics": {
      "AP": 0.425,
      "AP50": 0.652,
      "AP75": 0.461,
      "per_category_AP": {
        "person": 0.523,
        "car": 0.612,
        ...
      }
    }
  },
  "location_results": [
    {
      "image_path": "path/to/image.jpg",
      "predicted_items": ["cat", "dog"],
      "detections": [
        {
          "item": "cat",
          "objects": [{
            "name": "cat",
            "coordinates": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
            "normalized_coordinates": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
          }]
        }
      ],
      "processing_time": 0.234
    }
  ]
}
```

### FiftyOne Dataset Fields

When viewing results in FiftyOne, each sample includes:

- `predicted_detections` - List of predicted bounding boxes with labels
- `ground_truth` - Original ground truth annotations
- `processing_time` - Time taken for inference
- `num_predictions` - Number of objects detected
- `has_detections` - Boolean flag for filtering

## Advanced Usage

### Custom Detection Parameters

Adjust model parameters for detection:

```bash
# Higher temperature for more diverse predictions
uv run python detect_locations.py --model "model_name" \
    --evaluation-results label_results.json \
    --temperature 0.7 \
    --output results.json

# Increase max tokens for complex scenes
uv run python detect_locations.py --model "model_name" \
    --evaluation-results label_results.json \
    --max-tokens 512 \
    --output results.json
```

### Debug Mode

Enable verbose logging and debug output:

```bash
# Debug ground truth extraction
uv run python detect_locations.py --model "model_name" \
    --evaluation-results label_results.json \
    --debug-gt \
    --verbose \
    --output results.json
```

## Tips for Best Results

1. **Start Small**: Test with `--max-samples 10` to verify everything works
2. **Check Prerequisites**: Ensure you have classification results from `evaluate_labels_only.py`
3. **Monitor Memory**: Object detection models can be memory-intensive
4. **Compare IoU Thresholds**: Check AP50 vs AP75 to understand localization quality

## Troubleshooting

### Common Issues

1. **Missing Bounding Boxes**: Some models may predict objects without locations
2. **Coordinate Format Issues**: Check if model outputs are in correct format
3. **Memory Errors**: Use a smaller model
4. **Slow Processing**: Object detection is computationally intensive

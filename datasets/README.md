# Dataset Management for VLM Evaluation

This directory contains tools and examples for preparing datasets for Vision-Language Model (VLM) evaluation. The primary tool converts popular computer vision dataset formats (COCO and YOLO) into FiftyOne format, which is required by the evaluation scripts.

## Overview

The VLM evaluation pipeline requires datasets to be in FiftyOne format. This provides:
- Unified interface for different dataset formats
- Efficient data loading and filtering
- Rich metadata support
- Easy visualization and exploration

## Supported Dataset Formats

### 1. COCO Format

COCO (Common Objects in Context) format uses:
- A JSON annotations file containing images, categories, and annotations
- A directory of image files
- Bounding boxes in `[x, y, width, height]` format

Example structure:
```
coco_dataset/
├── annotations.json
├── image1.jpg
├── image2.jpg
└── ...
```

### 2. YOLO Format

YOLO (You Only Look Once) format uses:
- A `data.yaml` configuration file
- Separate directories for images and labels
- One `.txt` file per image with normalized coordinates
- Bounding boxes in `[class_id, x_center, y_center, width, height]` format

Example structure:
```
yolo_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Quick Start

### Converting COCO Dataset

```bash
# Basic conversion
uv run python convert_dataset_to_fiftyone_format.py \
    --format coco \
    --coco-annotations path/to/annotations.json \
    --coco-images-dir path/to/images \
    --dataset-name my_coco_dataset

# Using the example dataset
uv run python convert_dataset_to_fiftyone_format.py \
    --format coco \
    --coco-annotations example_dataset_formats/coco_dataset/ann_file.json \
    --coco-images-dir example_dataset_formats/coco_dataset \
    --dataset-name example_coco_dataset \
    --overwrite
```

### Converting YOLO Dataset

```bash
# Basic conversion
uv run python convert_dataset_to_fiftyone_format.py \
    --format yolo \
    --yolo-data-yaml path/to/data.yaml \
    --dataset-name my_yolo_dataset

# Convert specific splits only
uv run python convert_dataset_to_fiftyone_format.py \
    --format yolo \
    --yolo-data-yaml path/to/data.yaml \
    --dataset-name my_yolo_dataset \
    --splits train val
```

## Command-Line Options

### Required Arguments

- `--format {coco,yolo}` - Input dataset format
- `--dataset-name NAME` - Name for the FiftyOne dataset (used to load it later)

### Format-Specific Arguments

#### COCO Format
- `--coco-annotations PATH` - Path to COCO annotations JSON file
- `--coco-images-dir PATH` - Path to directory containing images

#### YOLO Format
- `--yolo-data-yaml PATH` - Path to YOLO data.yaml configuration file
- `--splits {train,val,test}` - Which splits to convert (default: all available)

### Optional Arguments

- `--overwrite` - Replace existing dataset if it already exists
- `--export-coco PATH` - Export converted dataset back to COCO format for verification
- `--verbose` - Enable detailed logging

## Dataset Format Details

### COCO Annotations Format

The COCO JSON file should contain:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person"
    },
    {
      "id": 2,
      "name": "car"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 50, 80],  // [x, y, width, height]
      "area": 4000
    }
  ]
}
```

### YOLO data.yaml Format

The YOLO configuration file should contain:

```yaml
# Dataset paths (relative to data.yaml location)
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 2

# Class names
names: ['person', 'car']
```

YOLO label files (one per image) contain:
```
0 0.5 0.5 0.2 0.3    # class_id x_center y_center width height
1 0.3 0.7 0.15 0.25  # (all values normalized 0-1)
```

## Using Converted Datasets

Once converted, datasets can be used with the VLM evaluation tools:

```bash
# Evaluate VLM on your custom dataset
cd ../classification
uv run python evaluate_labels_only.py \
    --model "google/gemma-3-12b-it" \
    --dataset-name my_coco_dataset \
    --output results.json

# View results in FiftyOne
uv run python label_viewer.py \
    --results results.json \
    --source-dataset my_coco_dataset
```

## Example Datasets

The `example_dataset_formats/` directory contains minimal examples of both formats:

### COCO Example
- 4 sample images with bounding box annotations
- Categories: person, bicycle, car, motorcycle, airplane, bus, train, truck

### YOLO Example
- Same 4 images converted to YOLO format
- Demonstrates proper directory structure and label format

## Troubleshooting

### Common Issues

1. **"Dataset already exists"**
   - Use `--overwrite` flag to replace existing dataset
   - Or choose a different `--dataset-name`

2. **"Image not found"**
   - Check that image paths in annotations match actual files
   - For COCO: paths are relative to `--coco-images-dir`
   - For YOLO: paths are relative to data.yaml location

3. **"No annotations found"**
   - Verify annotation format matches expected structure
   - Check category IDs match between annotations and categories

4. **Memory issues with large datasets**
   - The converter processes images in batches
   - Consider converting splits separately for very large datasets

### Validation

To verify your conversion:

1. **Check dataset info** - The script prints summary statistics after conversion
2. **Export to COCO** - Use `--export-coco` to re-export and compare
3. **View in FiftyOne** - Load and explore the dataset:

```python
import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("my_dataset_name")

# Print info
print(dataset)
print(f"Classes: {dataset.distinct('ground_truth.detections.label')}")
print(f"Samples: {len(dataset)}")

# Launch viewer
session = fo.launch_app(dataset)
```

## Best Practices

1. **Naming Conventions**
   - Use descriptive dataset names (e.g., `coco2017_val`, `custom_products_v2`)
   - Include version numbers for dataset iterations

2. **Data Quality**
   - Verify all images are accessible before conversion
   - Check that annotations are complete and accurate
   - Remove or fix corrupted images

3. **Organization**
   - Keep original data separate from converted datasets
   - Document any preprocessing or filtering applied
   - Maintain consistent category names across datasets

4. **Performance**
   - For large datasets, convert and evaluate on a subset first
   - Use `--splits val` to convert only validation data for testing
   - FiftyOne datasets are persistent - no need to reconvert

## Integration with VLM Evaluation

The converted FiftyOne datasets integrate seamlessly with the classification tools:

1. **Consistent Ground Truth** - Bounding boxes are converted to label lists for classification
2. **Metadata Preservation** - Image dimensions, paths, and other metadata are maintained
3. **Efficient Loading** - FiftyOne's backend enables fast data access during evaluation
4. **Hallucination Detection** - Ground truth labels enable automatic detection of model hallucinations


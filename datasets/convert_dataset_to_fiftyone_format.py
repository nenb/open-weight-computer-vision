#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

import fiftyone as fo
import fiftyone.types as fot
from tqdm import tqdm


class COCOToFiftyOne:
    """Convert COCO format dataset to FiftyOne format."""

    def __init__(self, annotations_path: str, images_dir: str):
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)

        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Create image mapping
        self.images = {img['id']: img for img in self.coco_data['images']}

        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

    def convert_to_fiftyone(self, dataset_name: str) -> fo.Dataset:
        """Convert COCO dataset to FiftyOne format."""
        # Create dataset
        dataset = fo.Dataset(name=dataset_name, persistent=True)

        # Process each image
        samples = []
        for img_id, img_info in tqdm(self.images.items(), desc="Converting COCO images"):
            # Get image path
            img_path = self.images_dir / img_info['file_name']
            if not img_path.exists():
                logging.warning(f"Image not found: {img_path}")
                continue

            # Create sample
            sample = fo.Sample(filepath=str(img_path))

            # Get annotations for this image
            annotations = self.image_annotations.get(img_id, [])

            # Convert to FiftyOne detections
            detections = []
            for ann in annotations:
                # Get bbox in COCO format [x, y, width, height]
                x, y, w, h = ann['bbox']

                # Convert to relative coordinates
                img_width = img_info['width']
                img_height = img_info['height']

                # FiftyOne expects [top-left-x, top-left-y, width, height] in relative coords
                rel_x = x / img_width
                rel_y = y / img_height
                rel_w = w / img_width
                rel_h = h / img_height

                # Create detection
                detection = fo.Detection(
                    label=self.categories[ann['category_id']],
                    bounding_box=[rel_x, rel_y, rel_w, rel_h]
                )
                detections.append(detection)

            # Add detections to sample
            sample["ground_truth"] = fo.Detections(detections=detections)
            samples.append(sample)

        # Add samples to dataset
        dataset.add_samples(samples)

        return dataset


class YOLOToFiftyOne:
    """Convert YOLO format dataset to FiftyOne format."""

    def __init__(self, data_yaml_path: str):
        self.data_yaml_path = Path(data_yaml_path)
        self.base_dir = self.data_yaml_path.parent

        # Load YOLO data.yaml
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        # Get class names
        self.class_names = self.data_config.get('names', {})
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}

        # Get paths
        self.train_path = self._resolve_path(self.data_config.get('train', ''))
        self.val_path = self._resolve_path(self.data_config.get('val', ''))
        self.test_path = self._resolve_path(self.data_config.get('test', ''))

    def _resolve_path(self, path: str) -> Optional[Path]:
        """Resolve path relative to data.yaml location."""
        if not path:
            return None

        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path

        return path if path.exists() else None

    def _parse_yolo_label(self, label_line: str, img_width: int, img_height: int) -> fo.Detection:
        """Parse a single YOLO label line and convert to FiftyOne detection."""
        parts = label_line.strip().split()
        class_id = int(parts[0])

        # YOLO format: [class_id, x_center, y_center, width, height] (all relative)
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert to top-left corner format for FiftyOne
        x_tl = x_center - width / 2
        y_tl = y_center - height / 2

        # Create detection
        detection = fo.Detection(
            label=self.class_names.get(class_id, f"class_{class_id}"),
            bounding_box=[x_tl, y_tl, width, height]
        )

        return detection

    def _process_split(self, split_path: Path, split_name: str) -> List[fo.Sample]:
        """Process a single split (train/val/test) of YOLO dataset."""
        samples = []

        # Get images and labels directories
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.exists():
            # Try alternative structure where split_path itself contains images
            images_dir = split_path
            labels_dir = split_path.parent / "labels"

        # Process all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        for img_path in tqdm(image_files, desc=f"Converting {split_name} images"):
            # Get corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"

            # Create sample
            sample = fo.Sample(filepath=str(img_path))

            # Load detections if label file exists
            detections = []
            if label_path.exists():
                # Get image dimensions (needed for some YOLO variants)
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                # Read label file
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            detection = self._parse_yolo_label(line, img_width, img_height)
                            detections.append(detection)

            # Add detections to sample
            sample["ground_truth"] = fo.Detections(detections=detections)
            samples.append(sample)

        return samples

    def convert_to_fiftyone(self, dataset_name: str, splits: None | List[str] = None) -> fo.Dataset:
        """Convert YOLO dataset to FiftyOne format."""
        # Create dataset
        dataset = fo.Dataset(name=dataset_name, persistent=True)

        # Default splits
        if splits is None:
            splits = []
            if self.train_path:
                splits.append('train')
            if self.val_path:
                splits.append('val')
            if self.test_path:
                splits.append('test')

        # Process each split
        all_samples = []
        for split in splits:
            if split == 'train' and self.train_path:
                samples = self._process_split(self.train_path, 'train')
                for sample in samples:
                    sample.tags.append('train')
                all_samples.extend(samples)
            elif split == 'val' and self.val_path:
                samples = self._process_split(self.val_path, 'validation')
                for sample in samples:
                    sample.tags.append('validation')
                all_samples.extend(samples)
            elif split == 'test' and self.test_path:
                samples = self._process_split(self.test_path, 'test')
                for sample in samples:
                    sample.tags.append('test')
                all_samples.extend(samples)

        # Add samples to dataset
        dataset.add_samples(all_samples)

        return dataset


def export_to_coco_format(dataset: fo.Dataset, output_path: str):
    """Export FiftyOne dataset to COCO format for verification."""
    # Export to COCO format
    dataset.export(
        export_dir=output_path,
        dataset_type=fot.COCODetectionDataset,
        label_field="ground_truth"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert COCO or YOLO dataset to FiftyOne format")

    # Input format
    parser.add_argument("--format", choices=["coco", "yolo"], required=True,
                       help="Input dataset format")

    # COCO specific arguments
    parser.add_argument("--coco-annotations",
                       help="Path to COCO annotations JSON file")
    parser.add_argument("--coco-images-dir",
                       help="Path to COCO images directory")

    # YOLO specific arguments
    parser.add_argument("--yolo-data-yaml",
                       help="Path to YOLO data.yaml file")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                       help="Which splits to convert (default: all available)")

    # Output configuration
    parser.add_argument("--dataset-name", required=True,
                       help="Name for the FiftyOne dataset")
    parser.add_argument("--export-coco",
                       help="Optional: Export converted dataset to COCO format for verification")

    # Other options
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing dataset if it exists")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Check if dataset already exists
    if fo.dataset_exists(args.dataset_name):
        if args.overwrite:
            logger.info(f"Deleting existing dataset: {args.dataset_name}")
            fo.delete_dataset(args.dataset_name)
        else:
            logger.error(f"Dataset '{args.dataset_name}' already exists. Use --overwrite to replace it.")
            return

    # Convert based on format
    if args.format == "coco":
        if not args.coco_annotations or not args.coco_images_dir:
            parser.error("COCO format requires --coco-annotations and --coco-images-dir")

        logger.info("Converting COCO dataset to FiftyOne format...")
        converter = COCOToFiftyOne(args.coco_annotations, args.coco_images_dir)
        dataset = converter.convert_to_fiftyone(args.dataset_name)

    elif args.format == "yolo":
        if not args.yolo_data_yaml:
            parser.error("YOLO format requires --yolo-data-yaml")

        logger.info("Converting YOLO dataset to FiftyOne format...")
        converter = YOLOToFiftyOne(args.yolo_data_yaml)
        dataset = converter.convert_to_fiftyone(args.dataset_name, args.splits)

    # Print dataset info
    logger.info(f"Conversion complete!")
    print(f"\nDataset Info:")
    print(f"Name: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.distinct('ground_truth.detections.label')}")
    print(f"Persistent: {dataset.persistent}")

    # Optional: Export to COCO format for verification
    if args.export_coco:
        logger.info(f"Exporting to COCO format at: {args.export_coco}")
        export_to_coco_format(dataset, args.export_coco)

    print(f"\nDataset '{args.dataset_name}' is ready for use with evaluate_labels_only.py")
    print(f"To load in FiftyOne: dataset = fo.load_dataset('{args.dataset_name}')")


if __name__ == "__main__":
    main()

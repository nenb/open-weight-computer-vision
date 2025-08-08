#!/usr/bin/env python3

import argparse
import json
import logging
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fiftyone as fo
import fiftyone.zoo as foz
import mlx.core as mx
import mlx_vlm
from tqdm import tqdm
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os


class LocationDetector:
    def __init__(self, model_path: str, max_tokens: int = 256, temperature: float = 0.0):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model, self.processor = mlx_vlm.load(model_path)
        self.config = mlx_vlm.utils.load_config(model_path)
        self.failed_detections = []

    def _generate_detection_prompt(self, item_name: str) -> str:
        return f"detect {item_name}"

    def detect_item_location(self, image_path: str, item_name: str) -> Tuple[str, Optional[str]]:
        try:
            prompt = self._generate_detection_prompt(item_name)
            prompt_template = mlx_vlm.prompt_utils.apply_chat_template(
                self.processor, self.config, prompt, num_images=1
            )

            result = mlx_vlm.generate(
                self.model,
                self.processor,
                prompt_template,
                [image_path],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False
            )

            return result.text, None

        except Exception as e:
            error_msg = f"Failed to detect {item_name} in {image_path}: {str(e)}"
            return "", error_msg

    def clear_cache(self):
        mx.clear_cache()


class LocationParser:
    # Regular expression to match PaliGemma location format
    _SEGMENT_DETECT_RE = re.compile(
        r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([a-zA-Z\s]+)'
    )

    @staticmethod
    def extract_objs(text: str, image_path: str) -> List[Dict]:
        """Extract objects and their locations from PaliGemma model output."""
        objects = []

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            # Fallback dimensions if image can't be opened
            width, height = 1024, 1024

        # Find all matches in the text
        matches = LocationParser._SEGMENT_DETECT_RE.findall(text)

        for match in matches:
            # Extract the four coordinate values and the object name
            y1_str, x1_str, y2_str, x2_str, obj_name = match

            # Convert to integers (these are the raw coordinate values)
            gs = [int(y1_str), int(x1_str), int(y2_str), int(x2_str)]

            # Normalize coordinates by dividing by 1024 (PaliGemma coordinate space)
            y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]

            # Scale to image pixels
            y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))

            # Create object detection result
            obj_dict = {
                "name": obj_name.strip(),
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "format": "bounding_box"
                },
                "normalized_coordinates": {
                    "x1": x1 / width,
                    "y1": y1 / height,
                    "x2": x2 / width,
                    "y2": y2 / height,
                    "format": "normalized_bounding_box"
                },
                "detection_type": "bounding_box",
                "raw_coordinates": gs
            }

            objects.append(obj_dict)

        return objects

    @staticmethod
    def parse_location_response(response_text: str, item_name: str, image_path: str = None) -> Dict:
        """Parse PaliGemma location detection response with proper coordinate conversion."""

        result = {
            "item_name": item_name,
            "raw_response": response_text,
            "coordinates": None,
            "detection_confidence": 1.0,
            "detection_type": None,
            "objects": []
        }

        # Try the new PaliGemma format extraction
        objects = LocationParser.extract_objs(response_text, image_path)
        if objects:
            result["objects"] = objects
            # For backward compatibility, set coordinates to the first detected object
            if objects:
                first_obj = objects[0]
                result["coordinates"] = first_obj["coordinates"]
                result["detection_type"] = first_obj["detection_type"]
            return result


class DatasetManager:
    def __init__(self, dataset_name: str = "coco-2017", split: str = "validation", debug: bool = False):
        self.dataset_name = dataset_name
        self.split = split
        self.debug = debug
        self.dataset = self.load_dataset()

    def load_dataset(self) -> fo.Dataset:
        try:
            dataset = fo.load_dataset(f"{self.dataset_name}-{self.split}")
        except ValueError:
            dataset = foz.load_zoo_dataset(
                self.dataset_name,
                split=self.split,
                dataset_name=f"{self.dataset_name}-{self.split}"
            )
        return dataset

    def get_sample_by_path(self, image_path: str) -> Optional[fo.Sample]:
        """Get FiftyOne sample by image path"""
        try:
            if self.debug:
                print(f"DEBUG: Looking for sample with path: {image_path}")
                print(f"DEBUG: Dataset has {len(self.dataset)} samples")

            # Try exact match first using proper FiftyOne syntax
            view = self.dataset.match({"filepath": image_path})
            if len(view) > 0:
                sample = view.first()
                if self.debug:
                    print(f"DEBUG: Found exact match for {image_path}")
                return sample

            # If no exact match, try to find by filename
            filename = Path(image_path).name
            if self.debug:
                print(f"DEBUG: Trying filename match for: {filename}")

                # Get a few sample filepaths to see the format
                sample_paths = [sample.filepath for sample in self.dataset.take(3)]
                print(f"DEBUG: Sample dataset paths: {sample_paths}")

            # Search by filename ending
            for sample in self.dataset:
                if sample.filepath.endswith(filename):
                    if self.debug:
                        print(f"DEBUG: Found filename match: {sample.filepath}")
                    return sample

            if self.debug:
                print(f"DEBUG: No sample found for {image_path} or {filename}")
            return None
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Error in get_sample_by_path: {e}")
                print(f"DEBUG: Trying alternative approach...")

            # Alternative approach: iterate through dataset
            try:
                filename = Path(image_path).name
                for sample in self.dataset:
                    if hasattr(sample, 'filepath'):
                        if sample.filepath == image_path or sample.filepath.endswith(filename):
                            if self.debug:
                                print(f"DEBUG: Found sample via iteration: {sample.filepath}")
                            return sample

                if self.debug:
                    print(f"DEBUG: No sample found via iteration")
                return None
            except Exception as e2:
                if self.debug:
                    print(f"DEBUG: Alternative approach also failed: {e2}")
                return None

    def extract_ground_truth_annotations(self, image_path: str) -> Dict:
        """Extract ground truth annotations for COCO evaluation"""
        sample = self.get_sample_by_path(image_path)

        # Debug output
        if self.debug:
            if sample is None:
                print(f"DEBUG: No sample found for path: {image_path}")
            elif not hasattr(sample, 'ground_truth') or sample.ground_truth is None:
                print(f"DEBUG: Sample has no ground_truth: {image_path}")
            elif not hasattr(sample.ground_truth, 'detections') or not sample.ground_truth.detections:
                print(f"DEBUG: Sample ground_truth has no detections: {image_path}")
            else:
                print(f"DEBUG: Found {len(sample.ground_truth.detections)} detections for: {image_path}")

        # Get image dimensions - try multiple methods
        width, height = 1024, 1024  # Default fallback
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Could not open image {image_path}: {e}")
            # If we can't open the image, try to get dimensions from sample
            if sample and hasattr(sample, 'metadata'):
                width = sample.metadata.get('width', 1024)
                height = sample.metadata.get('height', 1024)

        annotations = []
        categories = set()

        # Only process if we have a sample with ground truth
        if sample and sample.ground_truth and hasattr(sample.ground_truth, 'detections') and sample.ground_truth.detections:
            if self.debug:
                print(f"DEBUG: Processing {len(sample.ground_truth.detections)} ground truth detections")
            for i, detection in enumerate(sample.ground_truth.detections):
                try:
                    # Convert normalized coordinates to pixel coordinates
                    bbox = detection.bounding_box
                    x_norm, y_norm, w_norm, h_norm = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Convert to COCO format (x, y, width, height) in pixels
                    x = x_norm * width
                    y = y_norm * height
                    w = w_norm * width
                    h = h_norm * height

                    annotation = {
                        "id": len(annotations),  # Unique annotation ID
                        "category_name": detection.label,
                        "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
                        "bbox_normalized": [x_norm, y_norm, w_norm, h_norm],
                        "area": w * h,
                        "iscrowd": 0,
                        "confidence": getattr(detection, 'confidence', 1.0)
                    }

                    annotations.append(annotation)
                    categories.add(detection.label)
                    if self.debug:
                        print(f"DEBUG: Added annotation {i}: {detection.label} at {bbox}")
                except Exception as e:
                    if self.debug:
                        print(f"DEBUG: Error processing detection {i}: {e}")
        else:
            if self.debug:
                print(f"DEBUG: No valid ground truth found for {image_path}")

        image_info = {
            "id": sample.id if sample else hash(image_path) % (10**8),
            "file_name": Path(image_path).name,
            "width": int(width),
            "height": int(height),
            "path": image_path
        }

        if self.debug:
            print(f"DEBUG: Returning {len(annotations)} annotations for {image_path}")
        return {
            "annotations": annotations,
            "image_info": image_info,
            "categories": list(categories)
        }


class LocationEvaluator:
    def __init__(self, detector: LocationDetector, dataset_manager: DatasetManager, max_retries: int = 3):
        self.detector = detector
        self.dataset_manager = dataset_manager
        self.max_retries = max_retries
        self.failed_detections = []

    def evaluate_image_locations(self, image_path: str, predicted_items: List[str]) -> Dict:
        """Detect locations for all predicted items in an image"""
        start_time = time.time()

        detections = []
        label_and_bbox_positive_predictions = set()

        for item in predicted_items:
            detection_result = self._detect_with_retry(image_path, item)
            if detection_result:
                detections.append(detection_result)
                label_and_bbox_positive_predictions.add(item)

        # Extract ground truth annotations for COCO evaluation
        ground_truth = self.dataset_manager.extract_ground_truth_annotations(image_path)

        processing_time = time.time() - start_time

        return {
            "image_path": image_path,
            "predicted_items": predicted_items,
            "labelled_prediction_without_bbox": list(set(predicted_items) - label_and_bbox_positive_predictions),
            "detections": detections,
            "total_detections": len(detections),
            "ground_truth": ground_truth,
            "processing_time": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _detect_with_retry(self, image_path: str, item_name: str) -> Optional[Dict]:
        """Detect item location with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response_text, error = self.detector.detect_item_location(image_path, item_name)

                if error:
                    if attempt == self.max_retries - 1:
                        self.failed_detections.append({
                            "image_path": image_path,
                            "item_name": item_name,
                            "error": error,
                            "attempts": attempt + 1
                        })
                        return None
                    else:
                        time.sleep(2 ** attempt)
                        continue

                # Parse the response with image path for proper coordinate conversion
                parsed_result = LocationParser.parse_location_response(response_text, item_name, image_path)
                return parsed_result

            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.failed_detections.append({
                        "image_path": image_path,
                        "item_name": item_name,
                        "error": str(e),
                        "attempts": attempt + 1
                    })
                    return None
                else:
                    time.sleep(2 ** attempt)

        return None


def convert_to_coco_format(location_results: List[Dict]) -> Tuple[Dict, Dict]:
    """Convert detection results to COCO format for evaluation"""

    # Create ground truth COCO format
    gt_images = []
    gt_annotations = []
    gt_categories = {}
    category_id_counter = 1

    # Create predictions COCO format
    pred_annotations = []

    for i, result in enumerate(location_results):
        try:
            ground_truth = result.get("ground_truth", {})
            image_info = ground_truth.get("image_info", {})

            # Validate image_info has required fields
            if not image_info or "width" not in image_info or "height" not in image_info:
                print(f"Warning: Missing width/height for image {i}: {result.get('image_path', 'unknown')}")
                print(f"image_info: {image_info}")
                # Skip this image if we can't get dimensions
                continue

            # Add image info to ground truth
            gt_images.append({
                "id": hash(result["image_path"]) % (10**8),  # Create consistent image ID
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "file_name": image_info.get("file_name", Path(result["image_path"]).name)
            })

            image_id = hash(result["image_path"]) % (10**8)

            # Process ground truth annotations
            annotations = ground_truth.get("annotations", [])
            for ann in annotations:
                category_name = ann.get("category_name", "unknown")

                # Create category mapping
                if category_name not in gt_categories:
                    gt_categories[category_name] = {
                        "id": category_id_counter,
                        "name": category_name,
                        "supercategory": "object"
                    }
                    category_id_counter += 1

                gt_annotations.append({
                    "id": len(gt_annotations),
                    "image_id": image_id,
                    "category_id": gt_categories[category_name]["id"],
                    "bbox": ann.get("bbox", [0, 0, 1, 1]),
                    "area": ann.get("area", 1),
                    "iscrowd": ann.get("iscrowd", 0)
                })

            # Process predicted detections
            detections = result.get("detections", [])
            for detection in detections:
                objects = detection.get("objects", [])
                if objects:
                    for obj in objects:
                        category_name = obj.get("name", "unknown")

                        # Add category if not exists
                        if category_name not in gt_categories:
                            gt_categories[category_name] = {
                                "id": category_id_counter,
                                "name": category_name,
                                "supercategory": "object"
                            }
                            category_id_counter += 1

                        # Convert coordinates to COCO bbox format [x, y, width, height]
                        coords = obj.get("coordinates", {})
                        if coords and "x1" in coords:
                            x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
                            width = x2 - x1
                            height = y2 - y1

                            pred_annotations.append({
                                "image_id": image_id,
                                "category_id": gt_categories[category_name]["id"],
                                "bbox": [x1, y1, width, height],
                                "score": detection.get("detection_confidence", 1.0)
                            })

        except Exception as e:
            print(f"Error processing result {i}: {e}")
            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            continue

    # Convert categories dict to list
    categories_list = list(gt_categories.values())

    # Create COCO ground truth format
    coco_gt = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": categories_list,
        "info": {
            "description": "Ground truth annotations from COCO dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "VLM Object Detection Evaluation",
            "date_created": datetime.now(timezone.utc).isoformat()
        }
    }

    # Create COCO predictions format
    coco_pred = pred_annotations

    return coco_gt, coco_pred


def compute_coco_metrics(coco_gt: Dict, coco_pred: List[Dict]) -> Dict:
    """Compute COCO evaluation metrics"""
    try:
        # Create temporary files for COCO evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(coco_gt, gt_file, indent=2)
            gt_file_path = gt_file.name

        # Initialize COCO objects
        coco_gt_obj = COCO(gt_file_path)

        if not coco_pred:
            # No predictions to evaluate
            os.unlink(gt_file_path)
            return {
                "AP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "APs": 0.0,
                "APm": 0.0,
                "APl": 0.0,
                "AR1": 0.0,
                "AR10": 0.0,
                "AR100": 0.0,
                "ARs": 0.0,
                "ARm": 0.0,
                "ARl": 0.0,
                "per_category_AP": {}
            }

        # Load predictions
        coco_dt = coco_gt_obj.loadRes(coco_pred)

        # Run evaluation
        coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "AP": float(coco_eval.stats[0]),      # Average Precision @ IoU=0.50:0.95
            "AP50": float(coco_eval.stats[1]),    # Average Precision @ IoU=0.50
            "AP75": float(coco_eval.stats[2]),    # Average Precision @ IoU=0.75
            "APs": float(coco_eval.stats[3]),     # Average Precision @ small objects
            "APm": float(coco_eval.stats[4]),     # Average Precision @ medium objects
            "APl": float(coco_eval.stats[5]),     # Average Precision @ large objects
            "AR1": float(coco_eval.stats[6]),     # Average Recall given 1 detection per image
            "AR10": float(coco_eval.stats[7]),    # Average Recall given 10 detections per image
            "AR100": float(coco_eval.stats[8]),   # Average Recall given 100 detections per image
            "ARs": float(coco_eval.stats[9]),     # Average Recall @ small objects
            "ARm": float(coco_eval.stats[10]),    # Average Recall @ medium objects
            "ARl": float(coco_eval.stats[11])     # Average Recall @ large objects
        }

        # Per-category metrics
        per_category_ap = {}
        if len(coco_gt["categories"]) > 0 and hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
            for cat_idx, cat_info in enumerate(coco_gt["categories"]):
                try:
                    # Get precision for this category (category_idx, IoU_idx, area_idx, maxDet_idx)
                    if cat_idx < coco_eval.eval['precision'].shape[2]:  # Check category dimension
                        precision = coco_eval.eval['precision'][:, :, cat_idx, 0, 2]  # All IoU, all area, maxDets=100
                        if precision.size > 0:
                            valid_precision = precision[precision > -1]
                            if len(valid_precision) > 0:
                                ap = np.mean(valid_precision)
                                per_category_ap[cat_info["name"]] = float(ap) if not np.isnan(ap) else 0.0
                            else:
                                per_category_ap[cat_info["name"]] = 0.0
                        else:
                            per_category_ap[cat_info["name"]] = 0.0
                    else:
                        per_category_ap[cat_info["name"]] = 0.0
                except (IndexError, KeyError):
                    per_category_ap[cat_info["name"]] = 0.0

        metrics["per_category_AP"] = per_category_ap

        # Clean up temporary file
        os.unlink(gt_file_path)

        return metrics

    except Exception as e:
        print(f"Error computing COCO metrics: {e}")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "ARs": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
            "per_category_AP": {},
            "error": str(e)
        }


def load_evaluation_results(results_path: str) -> Dict:
    """Load results from evaluate_labels_only.py"""
    with open(results_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, output_path: str):
    """Save location detection results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """Load existing results from checkpoint"""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data.get('location_results', [])
    return []


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    """Save checkpoint during processing"""
    checkpoint_data = {
        'location_results': results,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def compute_location_statistics(location_results: List[Dict]) -> Dict:
    """Compute statistics about location detection results"""
    total_images = len(location_results)
    total_items = sum(len(result["predicted_items"]) for result in location_results)
    total_detections = sum(result["total_detections"] for result in location_results)
    total_processing_time = sum(result["processing_time"] for result in location_results)

    # Detection success rate
    detection_rate = total_detections / total_items if total_items > 0 else 0.0

    # Detection type distribution
    detection_types = {}
    coordinate_formats = {}

    for result in location_results:
        for detection in result["detections"]:
            det_type = detection.get("detection_type", "unknown")
            detection_types[det_type] = detection_types.get(det_type, 0) + 1

            if detection.get("coordinates"):
                coord_format = detection["coordinates"].get("format", "unknown")
                coordinate_formats[coord_format] = coordinate_formats.get(coord_format, 0) + 1

    return {
        "total_images_processed": total_images,
        "total_items_predicted": total_items,
        "total_successful_detections": total_detections,
        "detection_success_rate": detection_rate,
        "average_processing_time_per_image": total_processing_time / total_images if total_images > 0 else 0.0,
        "detection_type_distribution": detection_types,
        "coordinate_format_distribution": coordinate_formats
    }


def main():
    parser = argparse.ArgumentParser(description="Detect item locations using PaliGemma based on evaluate_labels_only.py results")

    # Required arguments
    parser.add_argument("--model", required=True,
                       help="Path to PaliGemma model or HuggingFace model ID")
    parser.add_argument("--evaluation-results", required=True,
                       help="Path to JSON file with results from evaluate_labels_only.py")
    parser.add_argument("--output", required=True,
                       help="Output JSON file path for location detection results")

    # Optional arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum tokens for model generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for model generation")
    parser.add_argument("--resume-from", default=None,
                       help="Resume processing from checkpoint file")
    parser.add_argument("--debug-gt", action="store_true",
                       help="Enable debug output for ground truth extraction")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load evaluation results
    logger.info(f"Loading evaluation results from {args.evaluation_results}")
    try:
        eval_results = load_evaluation_results(args.evaluation_results)
    except FileNotFoundError:
        logger.error(f"Evaluation results file not found: {args.evaluation_results}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in evaluation results file: {args.evaluation_results}")
        return

    # Extract sample results
    sample_results = eval_results.get("sample_results", [])
    if not sample_results:
        logger.error("No sample results found in evaluation file")
        return

    logger.info(f"Found {len(sample_results)} samples in evaluation results")

    # Limit samples if requested
    if args.max_samples:
        sample_results = sample_results[:args.max_samples]
        logger.info(f"Limited to {len(sample_results)} samples")

    # Initialize components
    logger.info("Loading PaliGemma model...")
    detector = LocationDetector(
        args.model,
        args.max_tokens,
        args.temperature
    )

    dataset_manager = DatasetManager(debug=args.debug_gt)
    evaluator = LocationEvaluator(detector, dataset_manager)

    # Load existing results if resuming
    location_results = []
    processed_paths = set()

    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        location_results = load_checkpoint(args.resume_from)
        processed_paths = {result["image_path"] for result in location_results}
        logger.info(f"Found {len(location_results)} previously processed samples")

    # Process samples
    checkpoint_path = args.output.replace('.json', '_checkpoint.json')

    try:
        for i, sample_result in enumerate(tqdm(sample_results, desc="Processing locations")):
            image_path = sample_result["image_path"]

            # Skip if already processed
            if image_path in processed_paths:
                continue

            # Get predicted items from evaluation results
            predicted_items = sample_result.get("predicted_labels", [])

            if not predicted_items:
                logger.debug(f"No predicted items for {image_path}, skipping")
                continue

            # Detect locations for all predicted items
            result = evaluator.evaluate_image_locations(image_path, predicted_items)
            location_results.append(result)

            # Periodic cleanup and checkpointing
            if (i + 1) % 50 == 0:
                detector.clear_cache()
                save_checkpoint(location_results, checkpoint_path)
                logger.info(f"Processed {i + 1} samples, saved checkpoint")

    except KeyboardInterrupt:
        logger.info("Location detection interrupted by user")
        save_checkpoint(location_results, checkpoint_path)

    # Compute statistics
    logger.info("Computing location detection statistics...")
    location_stats = compute_location_statistics(location_results)

    # Compute COCO metrics
    logger.info("Computing COCO evaluation metrics...")
    try:
        coco_gt, coco_pred = convert_to_coco_format(location_results)
        coco_metrics = compute_coco_metrics(coco_gt, coco_pred)
        logger.info(f"COCO AP: {coco_metrics['AP']:.3f}, AP50: {coco_metrics['AP50']:.3f}")

    except Exception as e:
        logger.error(f"Failed to compute COCO metrics: {e}")
        coco_metrics = {"error": str(e)}
        coco_gt, coco_pred = {}, []

    # Compile final results
    final_results = {
        "detection_config": {
            "model_path": args.model,
            "source_evaluation_file": args.evaluation_results,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "total_samples_processed": len(location_results),
            "detection_date": datetime.now(timezone.utc).isoformat()
        },
        "location_statistics": location_stats,
        "coco_evaluation": {
            "metrics": coco_metrics,
            "ground_truth_format": {
                "total_images": len(coco_gt.get("images", [])),
                "total_annotations": len(coco_gt.get("annotations", [])),
                "total_categories": len(coco_gt.get("categories", []))
            },
            "predictions_format": {
                "total_predictions": len(coco_pred)
            }
        },
        "processing_stats": {
            "total_time": sum(r["processing_time"] for r in location_results),
            "avg_time_per_image": location_stats["average_processing_time_per_image"],
            "failed_detections": len(evaluator.failed_detections),
            "failure_rate": len(evaluator.failed_detections) / (location_stats["total_items_predicted"] or 1)
        },
        "failed_detections": evaluator.failed_detections,
        "location_results": location_results
    }

    # Save results
    save_results(final_results, args.output)
    logger.info(f"Location detection results saved to {args.output}")

    # Print summary
    print(f"\nLocation Detection Complete!")
    print(f"Images processed: {len(location_results)}")
    print(f"Items predicted: {location_stats['total_items_predicted']}")
    print(f"Successful detections: {location_stats['total_successful_detections']}")
    print(f"Detection success rate: {location_stats['detection_success_rate']:.3f}")
    print(f"Failed detections: {len(evaluator.failed_detections)}")

    # Print COCO metrics if available
    if "error" not in coco_metrics:
        print(f"\nCOCO Evaluation Metrics:")
        print(f"  Average Precision (AP) @ IoU=0.50:0.95: {coco_metrics['AP']:.3f}")
        print(f"  Average Precision (AP) @ IoU=0.50: {coco_metrics['AP50']:.3f}")
        print(f"  Average Precision (AP) @ IoU=0.75: {coco_metrics['AP75']:.3f}")
        print(f"  Average Recall (AR) @ 100 detections: {coco_metrics['AR100']:.3f}")
        print(f"  Ground truth annotations: {len(coco_gt.get('annotations', []))}")
        print(f"  Predicted detections: {len(coco_pred)}")
    else:
        print(f"\nCOCO Evaluation failed: {coco_metrics['error']}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

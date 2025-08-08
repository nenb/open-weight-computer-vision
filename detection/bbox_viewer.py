#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fiftyone as fo
import fiftyone.zoo as foz


class BoundingBoxViewer:
    def __init__(self, dataset_name: str = "coco-2017", split: str = "validation"):
        self.dataset_name = dataset_name
        self.split = split
        self.full_dataset_name = f"{self.dataset_name}-{self.split}"

    def load_or_create_dataset(self) -> fo.Dataset:
        try:
            dataset = fo.load_dataset(self.full_dataset_name)
            logging.info(f"Loaded existing dataset: {self.full_dataset_name}")
        except ValueError:
            dataset = foz.load_zoo_dataset(
                self.dataset_name,
                split=self.split,
                dataset_name=self.full_dataset_name
            )
            logging.info(f"Created new dataset from zoo: {self.full_dataset_name}")
        return dataset

    def create_bbox_dataset(self, location_results: Dict, dataset_name: str) -> fo.Dataset:
        base_dataset = self.load_or_create_dataset()

        # Create a new dataset for the bbox view
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        bbox_dataset = fo.Dataset(dataset_name)
        bbox_dataset.persistent = True

        # Create mapping from filepath to location results
        results_by_path = {}
        for result in location_results.get("location_results", []):
            results_by_path[result["image_path"]] = result

        logging.info(f"Adding {len(results_by_path)} analyzed samples with bounding boxes to dataset")

        # Add samples with location metadata and bounding boxes
        samples = []
        for sample in base_dataset:
            if sample.filepath in results_by_path:
                result = results_by_path[sample.filepath]

                # Create new sample with original data
                new_sample = fo.Sample(filepath=sample.filepath)

                # Copy ground truth if it exists
                if hasattr(sample, 'ground_truth') and sample.ground_truth:
                    new_sample["ground_truth"] = sample.ground_truth

                # Add location metadata
                new_sample["predicted_items"] = result["predicted_items"]
                new_sample["total_detections"] = result["total_detections"]
                new_sample["processing_time"] = result["processing_time"]
                new_sample["labelled_prediction_without_bbox"] = result["labelled_prediction_without_bbox"]
                new_sample["timestamp"] = result["timestamp"]

                # Create predicted detections field with bounding boxes
                predicted_detections = []
                for detection in result["detections"]:
                    # Only process detections that have valid coordinates and objects
                    if (detection.get("coordinates") and
                        detection.get("objects") and
                        detection.get("detection_type") == "bounding_box"):

                        for obj in detection["objects"]:
                            coords = obj["normalized_coordinates"]

                            # Create FiftyOne detection object
                            fo_detection = fo.Detection(
                                label=obj["name"],
                                bounding_box=[
                                    coords["x1"],  # top-left x (normalized)
                                    coords["y1"],  # top-left y (normalized)
                                    coords["x2"] - coords["x1"],  # width (normalized)
                                    coords["y2"] - coords["y1"]   # height (normalized)
                                ],
                                confidence=1.0,  # Default confidence
                                raw_response=detection.get("raw_response", ""),
                                raw_coordinates=obj.get("raw_coordinates", []),
                                pixel_coordinates={
                                    "x1": obj["coordinates"]["x1"],
                                    "y1": obj["coordinates"]["y1"],
                                    "x2": obj["coordinates"]["x2"],
                                    "y2": obj["coordinates"]["y2"]
                                }
                            )
                            predicted_detections.append(fo_detection)

                # Add detections to sample
                new_sample["predicted_detections"] = fo.Detections(detections=predicted_detections)

                # Add summary statistics
                new_sample["num_predicted_items"] = len(result["predicted_items"])
                new_sample["num_detected_boxes"] = len(predicted_detections)
                new_sample["detection_success_rate"] = len(predicted_detections) / len(result["predicted_items"]) if result["predicted_items"] else 0.0

                # Add detection type summary - ensure all keys are strings
                detection_types = {}
                for detection in result["detections"]:
                    det_type = detection.get("detection_type", "unknown")
                    # Convert to string to ensure FiftyOne compatibility
                    det_type_str = str(det_type) if det_type is not None else "unknown"
                    detection_types[det_type_str] = detection_types.get(det_type_str, 0) + 1

                new_sample["detection_type_counts"] = detection_types

                samples.append(new_sample)

        bbox_dataset.add_samples(samples)

        # Add evaluation metadata to dataset
        detection_config = location_results.get("detection_config", {})
        location_stats = location_results.get("location_statistics", {})

        bbox_dataset.info = {
            "description": f"COCO {self.split} dataset with VLM location detection results and bounding boxes",
            "model_path": detection_config.get("model_path", "unknown"),
            "source_evaluation_file": detection_config.get("source_evaluation_file", "unknown"),
            "detection_date": detection_config.get("detection_date", "unknown"),
            "total_samples": len(samples),
            "total_images_processed": location_stats.get("total_images_processed", 0),
            "total_items_predicted": location_stats.get("total_items_predicted", 0),
            "total_successful_detections": location_stats.get("total_successful_detections", 0),
            "detection_success_rate": location_stats.get("detection_success_rate", 0.0),
            "average_processing_time": location_stats.get("average_processing_time_per_image", 0.0)
        }

        return bbox_dataset


def load_location_results(results_path: str) -> Dict:
    with open(results_path, 'r') as f:
        return json.load(f)


def create_filtered_views(dataset: fo.Dataset) -> Dict[str, fo.DatasetView]:
    views = {}

    # Samples with successful detections
    views["with_detections"] = dataset.match(fo.ViewField("num_detected_boxes") > 0)

    # Samples with no detections
    views["no_detections"] = dataset.match(fo.ViewField("num_detected_boxes") == 0)

    # High detection success rate (>= 0.8)
    views["high_success_rate"] = dataset.match(fo.ViewField("detection_success_rate") >= 0.8)

    # Low detection success rate (< 0.3)
    views["low_success_rate"] = dataset.match(fo.ViewField("detection_success_rate") < 0.3)

    # Perfect detection (all predicted items detected)
    views["perfect_detection"] = dataset.match(fo.ViewField("detection_success_rate") == 1.0)

    # Samples with many detections (>= 5 boxes)
    views["many_detections"] = dataset.match(fo.ViewField("num_detected_boxes") >= 5)

    # Samples sorted by processing time (slowest first)
    views["slowest_processing"] = dataset.sort_by("processing_time", reverse=True)

    # Samples sorted by number of detections (most first)
    views["most_detections"] = dataset.sort_by("num_detected_boxes", reverse=True)

    return views


def print_dataset_summary(dataset: fo.Dataset):
    print(f"\nDataset Summary:")
    print(f"Name: {dataset.name}")
    print(f"Total samples: {len(dataset)}")

    if dataset.info:
        info = dataset.info
        print(f"Model: {info.get('model_path', 'N/A')}")
        print(f"Detection date: {info.get('detection_date', 'N/A')}")
        print(f"Source evaluation file: {info.get('source_evaluation_file', 'N/A')}")
        print(f"Total images processed: {info.get('total_images_processed', 0)}")
        print(f"Total items predicted: {info.get('total_items_predicted', 0)}")
        print(f"Total successful detections: {info.get('total_successful_detections', 0)}")
        print(f"Overall detection success rate: {info.get('detection_success_rate', 0.0):.3f}")
        print(f"Average processing time per image: {info.get('average_processing_time', 0.0):.2f}s")

    # Print sample-level statistics
    if len(dataset) > 0:
        avg_success_rate = dataset.mean("detection_success_rate")
        avg_num_boxes = dataset.mean("num_detected_boxes")
        avg_processing_time = dataset.mean("processing_time")
        perfect_detections = len(dataset.match(fo.ViewField("detection_success_rate") == 1.0))
        no_detections = len(dataset.match(fo.ViewField("num_detected_boxes") == 0))

        print(f"\nSample-level statistics:")
        print(f"Average detection success rate: {avg_success_rate:.3f}")
        print(f"Average bounding boxes per image: {avg_num_boxes:.1f}")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Perfect detections: {perfect_detections} ({perfect_detections/len(dataset)*100:.1f}%)")
        print(f"Images with no detections: {no_detections} ({no_detections/len(dataset)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="View COCO dataset with VLM location detection results and bounding boxes in FiftyOne")

    parser.add_argument("--results", required=True,
                       help="Path to JSON file containing location detection results from detect_locations.py")
    parser.add_argument("--dataset-name", default="coco-bbox-analysis",
                       help="Name for the FiftyOne dataset with bounding box results")
    parser.add_argument("--coco-split", default="validation", choices=["train", "validation", "test"],
                       help="COCO split to use")
    parser.add_argument("--port", type=int, default=5151,
                       help="Port for FiftyOne app")
    parser.add_argument("--no-launch", action="store_true",
                       help="Don't automatically launch the FiftyOne app")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--view", choices=["all", "with_detections", "no_detections",
                                          "high_success_rate", "low_success_rate", "perfect_detection",
                                          "many_detections", "slowest_processing", "most_detections"],
                       default="all",
                       help="Which view to display in FiftyOne")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load location detection results
    logger.info(f"Loading location detection results from {args.results}")
    try:
        location_results = load_location_results(args.results)
    except FileNotFoundError:
        logger.error(f"Results file not found: {args.results}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results file: {e}")
        return

    # Create bbox viewer
    viewer = BoundingBoxViewer(split=args.coco_split)

    # Create dataset with bounding box results
    logger.info(f"Creating dataset with bounding box results: {args.dataset_name}")
    dataset = viewer.create_bbox_dataset(location_results, args.dataset_name)

    # Print summary
    print_dataset_summary(dataset)

    # Create filtered views
    views = create_filtered_views(dataset)

    print(f"\nAvailable views:")
    for view_name, view in views.items():
        print(f"  {view_name}: {len(view)} samples")

    # Select view to display
    if args.view == "all":
        current_view = dataset
    else:
        current_view = views[args.view]
        print(f"\nDisplaying view: {args.view} ({len(current_view)} samples)")

    # Launch FiftyOne app
    if not args.no_launch:
        print(f"\nLaunching FiftyOne app on port {args.port}...")
        print("Available fields for filtering and analysis:")
        print("  - predicted_items: List of predicted item names")
        print("  - predicted_detections: Bounding box detections with labels")
        print("  - total_detections: Total number of successful detections")
        print("  - num_predicted_items: Number of items that were predicted")
        print("  - num_detected_boxes: Number of detected bounding boxes")
        print("  - detection_success_rate: Ratio of detected boxes to predicted items")
        print("  - detection_type_counts: Count of detection types per image")
        print("  - processing_time: Time taken to detect locations for sample")
        print("\nBounding box metadata (accessible via predicted_detections):")
        print("  - raw_response: Original model response text")
        print("  - raw_coordinates: Raw coordinate values from model")
        print("  - pixel_coordinates: Absolute pixel coordinates")

        session = fo.launch_app(current_view, port=args.port)
        session.wait()
    else:
        print(f"\nDataset created: {args.dataset_name}")
        print("Use 'fo.launch_app()' or the FiftyOne CLI to view the dataset")


if __name__ == "__main__":
    main()

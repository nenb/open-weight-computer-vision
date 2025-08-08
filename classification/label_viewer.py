#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fiftyone as fo
import fiftyone.zoo as foz


class DatasetViewer:
    def __init__(self, dataset_name: str = "coco-2017", split: str = "validation", custom_dataset: str = None):
        self.dataset_name = dataset_name
        self.split = split
        self.custom_dataset = custom_dataset
        self.full_dataset_name = f"{self.dataset_name}-{self.split}" if not custom_dataset else custom_dataset

    def load_or_create_dataset(self) -> fo.Dataset:
        # First try to load custom dataset if provided
        if self.custom_dataset:
            try:
                dataset = fo.load_dataset(self.custom_dataset)
                logging.info(f"Loaded custom dataset: {self.custom_dataset}")
                return dataset
            except ValueError:
                raise ValueError(f"Custom dataset '{self.custom_dataset}' not found. "
                               f"Please ensure it exists or run convert_dataset_to_fiftyone_format.py first.")

        # Otherwise, try default behavior
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

    def create_view_dataset(self, analysis_results: Dict, dataset_name: str) -> fo.Dataset:
        base_dataset = self.load_or_create_dataset()

        # Create a new dataset for the view
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        view_dataset = fo.Dataset(dataset_name)
        view_dataset.persistent = True

        # Create mapping from filepath to analysis results
        results_by_path = {}
        for result in analysis_results.get("sample_results", []):
            results_by_path[result["image_path"]] = result

        logging.info(f"Adding {len(results_by_path)} analyzed samples to view dataset")

        # Get per-category metrics to identify hallucinated categories
        per_category_metrics = analysis_results.get("dataset_metrics", {}).get("per_category", {})
        hallucinated_categories = {cat for cat, metrics in per_category_metrics.items() if metrics.get("support", 0) == 0}

        # Add samples with analysis metadata
        samples = []
        for sample in base_dataset:
            if sample.filepath in results_by_path:
                result = results_by_path[sample.filepath]

                # Create new sample with original data
                new_sample = fo.Sample(filepath=sample.filepath)

                # Copy ground truth if it exists
                if hasattr(sample, 'ground_truth') and sample.ground_truth:
                    new_sample["ground_truth"] = sample.ground_truth

                # Add analysis metadata
                new_sample["predicted_labels"] = result["predicted_labels"]
                new_sample["ground_truth_labels"] = result["ground_truth_labels"]
                new_sample["model_response"] = result["model_response"]
                new_sample["processing_time"] = result["processing_time"]
                new_sample["timestamp"] = result["timestamp"]

                # Add metrics
                metrics = result["metrics"]
                new_sample["precision"] = metrics["precision"]
                new_sample["recall"] = metrics["recall"]
                new_sample["f1_score"] = metrics["f1_score"]
                new_sample["true_positives"] = metrics["true_positives"]
                new_sample["false_positives"] = metrics["false_positives"]
                new_sample["false_negatives"] = metrics["false_negatives"]

                # Add derived metrics
                new_sample["num_predicted"] = len(result["predicted_labels"])
                new_sample["num_ground_truth"] = len(result["ground_truth_labels"])
                new_sample["perfect_match"] = set(result["predicted_labels"]) == set(result["ground_truth_labels"])

                # Add hallucination detection
                hallucinated_labels = [label for label in result["predicted_labels"] if label in hallucinated_categories]
                new_sample["hallucinated_labels"] = hallucinated_labels
                new_sample["num_hallucinated"] = len(hallucinated_labels)
                new_sample["has_hallucination"] = len(hallucinated_labels) > 0

                samples.append(new_sample)

        view_dataset.add_samples(samples)

        # Add evaluation metadata to dataset
        eval_config = analysis_results.get("evaluation_config", {})
        dataset_metrics = analysis_results.get("dataset_metrics", {})

        dataset_desc = f"COCO {self.split} dataset with VLM label analysis results"
        if self.custom_dataset:
            dataset_desc = f"Custom dataset '{self.custom_dataset}' with VLM label analysis results"

        view_dataset.info = {
            "description": dataset_desc,
            "model_path": eval_config.get("model_path", "unknown"),
            "target_categories": eval_config.get("target_categories", []),
            "evaluation_date": eval_config.get("evaluation_date", "unknown"),
            "total_samples": len(samples),
            "overall_f1": dataset_metrics.get("overall", {}).get("f1_score", 0.0),
            "overall_precision": dataset_metrics.get("overall", {}).get("precision", 0.0),
            "overall_recall": dataset_metrics.get("overall", {}).get("recall", 0.0),
            "overall_accuracy": dataset_metrics.get("overall", {}).get("accuracy", 0.0)
        }

        return view_dataset


def load_analysis_results(results_path: str) -> Dict:
    with open(results_path, 'r') as f:
        return json.load(f)


def create_filtered_views(dataset: fo.Dataset) -> Dict[str, fo.DatasetView]:
    views = {}

    print(dataset)

    # Perfect matches (exact label match)
    views["perfect_matches"] = dataset.match(fo.ViewField("perfect_match") == True)

    # High performance samples (F1 > 0.8)
    views["high_f1"] = dataset.match(fo.ViewField("f1_score") > 0.8)

    # Low performance samples (F1 < 0.3)
    views["low_f1"] = dataset.match(fo.ViewField("f1_score") < 0.3)

    # Samples with false positives
    views["with_false_positives"] = dataset.match(fo.ViewField("false_positives") > 0)

    # Samples with false negatives
    views["with_false_negatives"] = dataset.match(fo.ViewField("false_negatives") > 0)

    # Samples with hallucinations
    views["with_hallucinations"] = dataset.match(fo.ViewField("has_hallucination") == True)

    # Samples sorted by processing time (slowest first)
    views["slowest_processing"] = dataset.sort_by("processing_time", reverse=True)

    return views


def print_dataset_summary(dataset: fo.Dataset):
    print(f"\nDataset Summary:")
    print(f"Name: {dataset.name}")
    print(f"Total samples: {len(dataset)}")

    if dataset.info:
        info = dataset.info
        print(f"Model: {info.get('model_path', 'N/A')}")
        print(f"Evaluation date: {info.get('evaluation_date', 'N/A')}")
        print(f"Target categories: {len(info.get('target_categories', []))}")
        print(f"Overall F1: {info.get('overall_f1', 0.0):.3f}")
        print(f"Overall Precision: {info.get('overall_precision', 0.0):.3f}")
        print(f"Overall Recall: {info.get('overall_recall', 0.0):.3f}")
        print(f"Overall Accuracy: {info.get('overall_accuracy', 0.0):.3f}")

    # Print some sample statistics
    if len(dataset) > 0:
        avg_f1 = dataset.mean("f1_score")
        avg_precision = dataset.mean("precision")
        avg_recall = dataset.mean("recall")
        perfect_matches = len(dataset.match(fo.ViewField("perfect_match") == True))

        print(f"\nSample-level statistics:")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Perfect matches: {perfect_matches} ({perfect_matches/len(dataset)*100:.1f}%)")

        # Hallucination statistics
        samples_with_hallucinations = len(dataset.match(fo.ViewField("has_hallucination") == True))
        if samples_with_hallucinations > 0:
            avg_hallucinations = dataset.mean("num_hallucinated")
            print(f"\nHallucination statistics:")
            print(f"Samples with hallucinations: {samples_with_hallucinations} ({samples_with_hallucinations/len(dataset)*100:.1f}%)")
            print(f"Average hallucinations per sample: {avg_hallucinations:.2f}")


def print_category_metrics(analysis_results: Dict):
    """Print per-category metrics with distinction between ground truth and hallucinated labels."""
    per_category = analysis_results.get("dataset_metrics", {}).get("per_category", {})

    if not per_category:
        return

    # Separate categories by support
    ground_truth_categories = {}
    hallucinated_categories = {}

    for category, metrics in per_category.items():
        if metrics.get("support", 0) > 0:
            ground_truth_categories[category] = metrics
        else:
            hallucinated_categories[category] = metrics

    # Print ground truth categories
    print("\n=== GROUND TRUTH CATEGORIES ===")
    print("Categories that actually appear in the dataset:")
    print(f"{'Category':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 60)

    for category, metrics in sorted(ground_truth_categories.items()):
        print(f"{category:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1']:<10.3f} {metrics['support']:<10}")

    # Print hallucinated categories if any
    if hallucinated_categories:
        print("\n=== HALLUCINATED CATEGORIES ===")
        print("Categories predicted by the model but NOT in ground truth (support=0):")
        print(f"{'Category':<20} {'False Positives':<15} {'Precision':<10}")
        print("-" * 45)

        # Calculate false positives from confusion matrix if available
        confusion_matrix = analysis_results.get("confusion_matrix", {})
        false_positives = confusion_matrix.get("false_positives", {})

        for category in sorted(hallucinated_categories.keys()):
            fp_count = false_positives.get(category, "N/A")
            precision = hallucinated_categories[category]['precision']
            print(f"{category:<20} {str(fp_count):<15} {precision:<10.3f}")

    # Summary statistics
    print(f"\nTotal categories in ground truth: {len(ground_truth_categories)}")
    print(f"Total hallucinated categories: {len(hallucinated_categories)}")

    if ground_truth_categories:
        avg_f1_gt = sum(m['f1'] for m in ground_truth_categories.values()) / len(ground_truth_categories)
        print(f"Average F1 for ground truth categories: {avg_f1_gt:.3f}")


def main():
    parser = argparse.ArgumentParser(description="View COCO or custom FiftyOne dataset with VLM label analysis results")

    parser.add_argument("--results", required=True,
                       help="Path to JSON file containing analysis results from evaluate_labels_only.py")
    parser.add_argument("--dataset-name", default="coco-vlm-analysis",
                       help="Name for the FiftyOne dataset with analysis results")
    parser.add_argument("--source-dataset", default=None,
                       help="Name of custom source FiftyOne dataset (if not using default COCO)")
    parser.add_argument("--coco-split", default="validation", choices=["train", "validation", "test"],
                       help="COCO split to use (ignored if --source-dataset is provided)")
    parser.add_argument("--port", type=int, default=5151,
                       help="Port for FiftyOne app")
    parser.add_argument("--no-launch", action="store_true",
                       help="Don't automatically launch the FiftyOne app")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--view", choices=["all", "perfect_matches", "high_f1", "low_f1",
                                          "with_false_positives", "with_false_negatives",
                                          "with_hallucinations", "slowest_processing"],
                       default="all",
                       help="Which view to display in FiftyOne")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load analysis results
    logger.info(f"Loading analysis results from {args.results}")
    try:
        analysis_results = load_analysis_results(args.results)
    except FileNotFoundError:
        logger.error(f"Results file not found: {args.results}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results file: {e}")
        return

    # Create dataset viewer
    viewer = DatasetViewer(split=args.coco_split, custom_dataset=args.source_dataset)

    # Create view dataset with analysis results
    logger.info(f"Creating dataset with analysis results: {args.dataset_name}")
    dataset = viewer.create_view_dataset(analysis_results, args.dataset_name)

    # Print summary
    print_dataset_summary(dataset)

    # Print category metrics with hallucination detection
    print_category_metrics(analysis_results)

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
        print("  - predicted_labels: List of predicted labels")
        print("  - ground_truth_labels: List of ground truth labels")
        print("  - precision, recall, f1_score: Per-sample metrics")
        print("  - true_positives, false_positives, false_negatives: Count metrics")
        print("  - perfect_match: Boolean indicating exact label match")
        print("  - processing_time: Time taken to analyze sample")
        print("  - num_predicted, num_ground_truth: Count of labels")
        print("  - hallucinated_labels: List of labels predicted but not in ground truth dataset")
        print("  - num_hallucinated: Count of hallucinated labels")
        print("  - has_hallucination: Boolean indicating if sample has any hallucinated labels")

        session = fo.launch_app(current_view, port=args.port)
        session.wait()
    else:
        print(f"\nDataset created: {args.dataset_name}")
        print("Use 'fo.launch_app()' or the FiftyOne CLI to view the dataset")


if __name__ == "__main__":
    main()

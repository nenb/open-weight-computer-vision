#!/usr/bin/env python3

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fiftyone as fo
import fiftyone.zoo as foz
import mlx.core as mx
import mlx_vlm
from tqdm import tqdm


class DatasetManager:
    def __init__(self, dataset_name: str = "coco-2017", split: str = "validation", custom_dataset: str = None):
        self.dataset_name = dataset_name
        self.split = split
        self.custom_dataset = custom_dataset
        self.dataset = self.load_dataset()

    def load_dataset(self) -> fo.Dataset:
        # First try to load custom dataset if provided
        if self.custom_dataset:
            try:
                dataset = fo.load_dataset(self.custom_dataset)
                return dataset
            except ValueError:
                raise ValueError(f"Custom dataset '{self.custom_dataset}' not found. "
                               f"Please ensure it exists or run convert_dataset_to_fiftyone_format.py first.")

        # Otherwise, try default behavior
        try:
            dataset = fo.load_dataset(f"{self.dataset_name}-{self.split}")
        except ValueError:
            dataset = foz.load_zoo_dataset(
                self.dataset_name,
                split=self.split,
                dataset_name=f"{self.dataset_name}-{self.split}"
            )
        return dataset

    def get_image_labels(self, sample: fo.Sample) -> List[str]:
        labels = []
        if sample.ground_truth:
            for detection in sample.ground_truth.detections:
                labels.append(detection.label.lower())
        return list(set(labels))


class MetricsCalculator:
    def compute_image_metrics(self, ground_truth: List[str], predictions: List[str]) -> Dict:
        gt_set = set(ground_truth)
        pred_set = set(predictions)

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    def compute_dataset_metrics(self, all_results: List[Dict]) -> Dict:
        total_tp = sum(r["metrics"]["true_positives"] for r in all_results)
        total_fp = sum(r["metrics"]["false_positives"] for r in all_results)
        total_fn = sum(r["metrics"]["false_negatives"] for r in all_results)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        # Per-category metrics
        per_category = {}
        category_stats = {}

        for result in all_results:
            gt_labels = result["ground_truth_labels"]
            pred_labels = result["predicted_labels"]

            for label in set(gt_labels + pred_labels):
                if label not in category_stats:
                    category_stats[label] = {"tp": 0, "fp": 0, "fn": 0}

                if label in gt_labels and label in pred_labels:
                    category_stats[label]["tp"] += 1
                elif label in pred_labels and label not in gt_labels:
                    category_stats[label]["fp"] += 1
                elif label in gt_labels and label not in pred_labels:
                    category_stats[label]["fn"] += 1

        for category, stats in category_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_category[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }

        accuracy = self._compute_accuracy(all_results)

        return {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1_score": overall_f1,
                "accuracy": accuracy
            },
            "per_category": per_category
        }

    def _compute_accuracy(self, all_results: List[Dict]) -> float:
        if not all_results:
            return 0.0

        correct = 0
        total = len(all_results)

        for result in all_results:
            gt_set = set(result["ground_truth_labels"])
            pred_set = set(result["predicted_labels"])
            if gt_set == pred_set:
                correct += 1

        return correct / total


class ModelEvaluator:
    def __init__(self, model_path: str, categories: List[str], max_tokens: int = 256, temperature: float = 0.0):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model, self.processor = mlx_vlm.load(model_path)
        self.config = mlx_vlm.utils.load_config(model_path)
        self.failed_samples = []
        self.prompt_template = mlx_vlm.prompt_utils.apply_chat_template(
                self.processor, self.config, self._generate_prompt(categories), num_images=1
            )

    def _generate_prompt(self, categories: List[str]) -> str:
        return f"Identify which items from the following list are contained in the image. Only return the items which are present. DO NOT return items that are not contained in the list. You must write your answer as a comma separated list. <LIST> {categories} </LIST>"

    def generate_prediction(self, image_path: str) -> Tuple[List[str], str, Optional[str]]:
        try:
            result = mlx_vlm.generate(
                self.model,
                self.processor,
                self.prompt_template,
                [image_path],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False
            )

            # Parse response
            predicted_labels = [label.strip() for label in result.text.split(",")]
            return predicted_labels, result.text, None

        except Exception as e:
            error_msg = f"Failed to generate prediction for {image_path}: {str(e)}"
            return "", None, error_msg

    def clear_cache(self):
        mx.clear_cache()


class RobustEvaluator:
    def __init__(self, model_evaluator: ModelEvaluator, metrics_calc: MetricsCalculator, max_retries: int = 3):
        self.model_evaluator = model_evaluator
        self.metrics_calc = metrics_calc
        self.max_retries = max_retries
        self.failed_samples = []

    def evaluate_sample_with_retry(self, sample: fo.Sample, dataset_manager: DatasetManager) -> Optional[Dict]:
        for attempt in range(self.max_retries):
            try:
                return self._evaluate_single_sample(sample, dataset_manager)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.failed_samples.append({
                        "sample_path": sample.filepath,
                        "error": str(e),
                        "attempts": attempt + 1
                    })
                    return None
                else:
                    time.sleep(2 ** attempt)
        return None

    def _evaluate_single_sample(self, sample: fo.Sample, dataset_manager: DatasetManager) -> Dict:
        start_time = time.time()

        # Extract ground truth labels
        ground_truth = dataset_manager.get_image_labels(sample)

        # Generate model prediction
        predicted_labels, model_response, error = self.model_evaluator.generate_prediction(sample.filepath)

        if error:
            raise Exception(error)

        # Compute metrics
        metrics = self.metrics_calc.compute_image_metrics(ground_truth, predicted_labels)

        processing_time = time.time() - start_time

        return {
            "image_path": sample.filepath,
            "ground_truth_labels": ground_truth,
            "predicted_labels": predicted_labels,
            "model_response": model_response,
            "metrics": metrics,
            "processing_time": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def save_results(results: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data.get('sample_results', [])
    return []


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    checkpoint_data = {
        'sample_results': results,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLX VLM model on COCO or custom FiftyOne dataset")

    # Model configuration
    parser.add_argument("--model", required=True,
                       help="Path to MLX VLM model or HuggingFace model ID")

    # Dataset configuration
    parser.add_argument("--dataset-name", default=None,
                       help="Name of custom FiftyOne dataset to use (if not using default COCO)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")

    # Output configuration
    parser.add_argument("--output", required=True,
                       help="Output JSON file path for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    # Model parameters
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum tokens for model generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for model generation")

    # Processing options
    parser.add_argument("--resume-from", default=None,
                       help="Resume evaluation from checkpoint file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize components
    logger.info("Loading dataset...")
    dataset_manager = DatasetManager(custom_dataset=args.dataset_name)
    dataset = dataset_manager.dataset
    categories = dataset.distinct("ground_truth.detections.label")
    if len(categories) == 0:
        logger.error("No categories found in ground truth dataset, exiting...")
        exit()

    if args.max_samples:
        dataset = dataset.limit(args.max_samples)

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    logger.info("Loading model...")
    model_evaluator = ModelEvaluator(
        args.model,
        categories,
        args.max_tokens,
        args.temperature
    )

    metrics_calc = MetricsCalculator()
    evaluator = RobustEvaluator(model_evaluator, metrics_calc)

    # Load existing results if resuming
    sample_results = []
    processed_paths = set()

    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        sample_results = load_checkpoint(args.resume_from)
        processed_paths = {r["image_path"] for r in sample_results}
        logger.info(f"Found {len(sample_results)} previously processed samples")

    # Process samples
    checkpoint_path = args.output.replace('.json', '_checkpoint.json')

    try:
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            # Skip if already processed
            if sample.filepath in processed_paths:
                continue

            result = evaluator.evaluate_sample_with_retry(sample, dataset_manager)

            if result:
                sample_results.append(result)

            # Periodic cleanup and checkpointing
            if (i + 1) % 100 == 0:
                model_evaluator.clear_cache()
                save_checkpoint(sample_results, checkpoint_path)
                logger.info(f"Processed {i + 1} samples, saved checkpoint")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        save_checkpoint(sample_results, checkpoint_path)

    # Compute final metrics
    logger.info("Computing final metrics...")
    dataset_metrics = metrics_calc.compute_dataset_metrics(sample_results)

    # Prepare confusion matrix data
    confusion_matrix = {"true_positives": {}, "false_positives": {}, "false_negatives": {}}
    for category in categories:
        confusion_matrix["true_positives"][category] = 0
        confusion_matrix["false_positives"][category] = 0
        confusion_matrix["false_negatives"][category] = 0

    for result in sample_results:
        gt_set = set(result["ground_truth_labels"])
        pred_set = set(result["predicted_labels"])

        for category in categories:
            if category in gt_set and category in pred_set:
                confusion_matrix["true_positives"][category] += 1
            elif category in pred_set and category not in gt_set:
                confusion_matrix["false_positives"][category] += 1
            elif category in gt_set and category not in pred_set:
                confusion_matrix["false_negatives"][category] += 1

    # Compile final results
    final_results = {
        "evaluation_config": {
            "model_path": args.model,
            "prompt_template": model_evaluator.prompt_template,
            "target_categories": categories,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "total_samples": len(sample_results),
            "evaluation_date": datetime.now(timezone.utc).isoformat()
        },
        "dataset_metrics": dataset_metrics,
        "confusion_matrix": confusion_matrix,
        "processing_stats": {
            "total_time": sum(r["processing_time"] for r in sample_results),
            "avg_time_per_image": sum(r["processing_time"] for r in sample_results) / len(sample_results) if sample_results else 0,
            "failed_predictions": len(evaluator.failed_samples),
            "error_rate": len(evaluator.failed_samples) / (len(sample_results) + len(evaluator.failed_samples)) if (len(sample_results) + len(evaluator.failed_samples)) > 0 else 0
        },
        "failed_samples": evaluator.failed_samples,
        "sample_results": sample_results
    }

    # Save results
    save_results(final_results, args.output)
    logger.info(f"Results saved to {args.output}")

    # Print summary
    print(f"\nEvaluation Complete!")
    print(f"Processed: {len(sample_results)} samples")
    print(f"Failed: {len(evaluator.failed_samples)} samples")
    print(f"Overall F1 Score: {dataset_metrics['overall']['f1_score']:.3f}")
    print(f"Overall Precision: {dataset_metrics['overall']['precision']:.3f}")
    print(f"Overall Recall: {dataset_metrics['overall']['recall']:.3f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

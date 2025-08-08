#!/usr/bin/env python3

import argparse
import json
import logging
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')  # Use non-interactive backend


class LabelResultsAnalyzer:
    def __init__(self, json_files: List[str]):
        self.results_data = {}
        self.load_results(json_files)

    def load_results(self, json_files: List[str]):
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract model name from path or config
                model_name = Path(file_path).stem
                if 'model_path' in data.get('evaluation_config', {}):
                    model_path = data['evaluation_config']['model_path']
                    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
                self.results_data[model_name] = data

    def analyze_single_file(self, model_name: str) -> Dict:
        data = self.results_data[model_name]
        metrics = data['dataset_metrics']
        sample_results = data.get('sample_results', [])

        # Basic statistics
        overall_metrics = metrics['overall']
        per_category = metrics.get('per_category', {})

        # Processing statistics
        processing_stats = data.get('processing_stats', {})

        # Error analysis
        confusion_matrix = data.get('confusion_matrix', {})

        # Response length analysis
        response_lengths = [len(result.get('model_response', '').split(','))
                          for result in sample_results]
        gt_lengths = [len(result.get('ground_truth_labels', []))
                     for result in sample_results]

        # Per-sample metrics distribution
        precisions = [result['metrics']['precision'] for result in sample_results
                     if 'metrics' in result]
        recalls = [result['metrics']['recall'] for result in sample_results
                  if 'metrics' in result]
        f1_scores = [result['metrics']['f1_score'] for result in sample_results
                    if 'metrics' in result]

        # Category performance ranking
        category_f1_scores = [(cat, stats['f1']) for cat, stats in per_category.items() if stats.get('support', 0) > 0]
        category_f1_scores.sort(key=lambda x: x[1], reverse=True)

        # Separate ground truth and hallucinated categories
        ground_truth_categories = {cat: stats for cat, stats in per_category.items()
                                 if stats.get('support', 0) > 0}
        hallucinated_categories = {cat: stats for cat, stats in per_category.items()
                                 if stats.get('support', 0) == 0}

        # False positive/negative analysis
        fp_by_category = defaultdict(int)
        fn_by_category = defaultdict(int)
        hallucination_count_per_sample = []

        for result in sample_results:
            gt_set = set(result.get('ground_truth_labels', []))
            pred_set = set(result.get('predicted_labels', []))

            # Count hallucinations in this sample
            hallucinations_in_sample = 0
            for pred in pred_set - gt_set:
                fp_by_category[pred] += 1
                if pred in hallucinated_categories:
                    hallucinations_in_sample += 1

            hallucination_count_per_sample.append(hallucinations_in_sample)

            for gt in gt_set - pred_set:
                fn_by_category[gt] += 1

        # Hallucination statistics
        samples_with_hallucinations = sum(1 for count in hallucination_count_per_sample if count > 0)
        avg_hallucinations_per_sample = np.mean(hallucination_count_per_sample) if hallucination_count_per_sample else 0

        return {
            'overall_metrics': overall_metrics,
            'processing_stats': processing_stats,
            'response_length_stats': {
                'mean_predictions': np.mean(response_lengths) if response_lengths else 0,
                'std_predictions': np.std(response_lengths) if response_lengths else 0,
                'mean_ground_truth': np.mean(gt_lengths) if gt_lengths else 0,
                'std_ground_truth': np.std(gt_lengths) if gt_lengths else 0,
            },
            'metric_distributions': {
                'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)} if precisions else {'mean': 0, 'std': 0},
                'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)} if recalls else {'mean': 0, 'std': 0},
                'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)} if f1_scores else {'mean': 0, 'std': 0},
            },
            'top_10_categories': category_f1_scores[:10],
            'bottom_10_categories': category_f1_scores[-10:],
            'most_false_positives': sorted(fp_by_category.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_false_negatives': sorted(fn_by_category.items(), key=lambda x: x[1], reverse=True)[:10],
            'total_categories': len(per_category) - len(hallucinated_categories),
            'total_samples': len(sample_results),
            'ground_truth_categories': ground_truth_categories,
            'hallucinated_categories': hallucinated_categories,
            'hallucination_stats': {
                'total_hallucinated_categories': len(hallucinated_categories),
                'samples_with_hallucinations': samples_with_hallucinations,
                'hallucination_rate': samples_with_hallucinations / len(sample_results) if sample_results else 0,
                'avg_hallucinations_per_sample': avg_hallucinations_per_sample,
                'hallucinated_category_list': sorted(hallucinated_categories.keys()),
                'hallucination_fp_counts': {cat: fp_by_category.get(cat, 0) for cat in hallucinated_categories}
            }
        }

    def compare_models(self) -> Dict:
        if len(self.results_data) < 2:
            return {}

        comparison = {
            'models': list(self.results_data.keys()),
            'overall_comparison': {},
            'category_comparison': defaultdict(dict),
            'processing_comparison': {},
            'hallucination_comparison': {},
        }

        # Overall metrics comparison
        for model_name, data in self.results_data.items():
            overall = data['dataset_metrics']['overall']
            comparison['overall_comparison'][model_name] = overall

            # Processing stats
            if 'processing_stats' in data:
                comparison['processing_comparison'][model_name] = {
                    'avg_time_per_image': data['processing_stats'].get('avg_time_per_image', 0),
                    'error_rate': data['processing_stats'].get('error_rate', 0),
                }

            # Hallucination stats
            analysis = self.analyze_single_file(model_name)
            hall_stats = analysis.get('hallucination_stats', {})
            comparison['hallucination_comparison'][model_name] = {
                'total_hallucinated_categories': hall_stats.get('total_hallucinated_categories', 0),
                'hallucination_rate': hall_stats.get('hallucination_rate', 0),
                'avg_hallucinations_per_sample': hall_stats.get('avg_hallucinations_per_sample', 0),
                'hallucinated_categories': hall_stats.get('hallucinated_category_list', [])
            }

        # Per-category comparison
        all_categories = set()
        for data in self.results_data.values():
            all_categories.update(data['dataset_metrics'].get('per_category', {}).keys())

        for category in all_categories:
            for model_name, data in self.results_data.items():
                per_cat = data['dataset_metrics'].get('per_category', {})
                if category in per_cat:
                    comparison['category_comparison'][category][model_name] = per_cat[category]['f1']

        # Find categories with biggest performance differences
        category_diffs = []
        for category, model_scores in comparison['category_comparison'].items():
            if len(model_scores) >= 2:
                scores = list(model_scores.values())
                diff = max(scores) - min(scores)
                category_diffs.append((category, diff, model_scores))

        category_diffs.sort(key=lambda x: x[1], reverse=True)
        comparison['largest_category_differences'] = category_diffs[:10]

        return comparison

    def generate_visualizations(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)

        # Single model visualizations
        for model_name in self.results_data:
            self._generate_single_model_plots(model_name, output_dir)

        # Comparison visualizations if multiple models
        if len(self.results_data) > 1:
            self._generate_comparison_plots(output_dir)

    def _generate_single_model_plots(self, model_name: str, output_dir: Path):
        data = self.results_data[model_name]
        analysis = self.analyze_single_file(model_name)

        # 1. Category F1 scores (top and bottom)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Top 10 categories
        top_cats = [cat for cat, _ in analysis['top_10_categories']]
        top_scores = [score for _, score in analysis['top_10_categories']]
        ax1.barh(top_cats, top_scores, color='green', alpha=0.7)
        ax1.set_xlabel('F1 Score')
        ax1.set_title(f'Top 10 Categories - {model_name}')
        ax1.set_xlim(0, 1)

        # Bottom 10 categories
        bottom_cats = [cat for cat, _ in analysis['bottom_10_categories']]
        bottom_scores = [score for _, score in analysis['bottom_10_categories']]
        ax2.barh(bottom_cats, bottom_scores, color='red', alpha=0.7)
        ax2.set_xlabel('F1 Score')
        ax2.set_title(f'Bottom 10 Categories - {model_name}')
        ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_category_performance.png', dpi=150)
        plt.close()

        # 2. Metric distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sample_results = data.get('sample_results', [])

        metrics_data = {
            'Precision': [r['metrics']['precision'] for r in sample_results if 'metrics' in r],
            'Recall': [r['metrics']['recall'] for r in sample_results if 'metrics' in r],
            'F1 Score': [r['metrics']['f1_score'] for r in sample_results if 'metrics' in r],
        }

        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            if values:
                axes[idx].hist(values, bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(metric_name)
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{metric_name} Distribution - {model_name}')
                axes[idx].axvline(np.mean(values), color='red', linestyle='--',
                                label=f'Mean: {np.mean(values):.3f}')
                axes[idx].legend()

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_metric_distributions.png', dpi=150)
        plt.close()

        # 3. False positives/negatives analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Most common false positives
        fp_cats = [cat for cat, _ in analysis['most_false_positives'][:10]]
        fp_counts = [count for _, count in analysis['most_false_positives'][:10]]
        ax1.barh(fp_cats, fp_counts, color='orange', alpha=0.7)
        ax1.set_xlabel('Count')
        ax1.set_title(f'Most Common False Positives - {model_name}')

        # Most common false negatives
        fn_cats = [cat for cat, _ in analysis['most_false_negatives'][:10]]
        fn_counts = [count for _, count in analysis['most_false_negatives'][:10]]
        ax2.barh(fn_cats, fn_counts, color='purple', alpha=0.7)
        ax2.set_xlabel('Count')
        ax2.set_title(f'Most Common False Negatives - {model_name}')

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_error_analysis.png', dpi=150)
        plt.close()

        # 4. Hallucination analysis
        hallucination_stats = analysis.get('hallucination_stats', {})
        if hallucination_stats.get('total_hallucinated_categories', 0) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Hallucinated categories with their false positive counts
            hallucinated_cats = hallucination_stats['hallucinated_category_list'][:10]
            hallucinated_counts = [hallucination_stats['hallucination_fp_counts'].get(cat, 0)
                                 for cat in hallucinated_cats]

            ax1.barh(hallucinated_cats, hallucinated_counts, color='red', alpha=0.7)
            ax1.set_xlabel('Number of False Positives')
            ax1.set_title(f'Hallucinated Categories (Not in Ground Truth) - {model_name}')

            # Hallucination distribution
            sample_results = data.get('sample_results', [])
            hallucination_counts = []
            for result in sample_results:
                gt_set = set(result.get('ground_truth_labels', []))
                pred_set = set(result.get('predicted_labels', []))
                hallucinated_in_sample = [p for p in pred_set - gt_set
                                        if p in hallucination_stats['hallucinated_category_list']]
                hallucination_counts.append(len(hallucinated_in_sample))

            # Create histogram
            max_hall = max(hallucination_counts) if hallucination_counts else 0
            bins = np.arange(0, max_hall + 2) - 0.5
            ax2.hist(hallucination_counts, bins=bins, alpha=0.7, color='red', edgecolor='black')
            ax2.set_xlabel('Number of Hallucinations per Sample')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title(f'Distribution of Hallucinations per Sample - {model_name}')
            ax2.set_xticks(range(0, max_hall + 1))

            # Add mean line
            mean_hall = np.mean(hallucination_counts)
            ax2.axvline(mean_hall, color='darkred', linestyle='--',
                       label=f'Mean: {mean_hall:.2f}')
            ax2.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_hallucination_analysis.png', dpi=150)
            plt.close()

    def _generate_comparison_plots(self, output_dir: Path):
        comparison = self.compare_models()

        # 1. Overall metrics comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        models = list(self.results_data.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [comparison['overall_comparison'][model].get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_overall.png', dpi=150)
        plt.close()

        # 2. Processing time comparison
        if comparison['processing_comparison']:
            fig, ax = plt.subplots(figsize=(10, 6))

            models = list(comparison['processing_comparison'].keys())
            times = [comparison['processing_comparison'][m]['avg_time_per_image']
                    for m in models]

            ax.bar(models, times, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Model')
            ax.set_ylabel('Average Time per Image (seconds)')
            ax.set_title('Processing Speed Comparison')

            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison_speed.png', dpi=150)
            plt.close()

        # 3. Category differences heatmap
        if len(comparison['largest_category_differences']) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))

            categories = [cat for cat, _, _ in comparison['largest_category_differences']]
            models = list(self.results_data.keys())

            matrix = np.zeros((len(categories), len(models)))
            for i, (cat, _, scores) in enumerate(comparison['largest_category_differences']):
                for j, model in enumerate(models):
                    matrix[i, j] = scores.get(model, 0)

            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(models)))
            ax.set_yticks(np.arange(len(categories)))
            ax.set_xticklabels(models)
            ax.set_yticklabels(categories)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add text annotations
            for i in range(len(categories)):
                for j in range(len(models)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)

            ax.set_title("Categories with Largest Performance Differences")
            fig.colorbar(im, ax=ax, label='F1 Score')

            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison_category_differences.png', dpi=150)
            plt.close()

        # 4. Hallucination comparison
        if comparison.get('hallucination_comparison'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            models = list(comparison['hallucination_comparison'].keys())

            # Hallucination rates
            hall_rates = [comparison['hallucination_comparison'][m]['hallucination_rate'] * 100
                         for m in models]
            ax1.bar(models, hall_rates, alpha=0.7, color='red', edgecolor='black')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Hallucination Rate (%)')
            ax1.set_title('Percentage of Samples with Hallucinations')
            ax1.set_ylim(0, max(hall_rates) * 1.2 if hall_rates else 100)

            # Average hallucinations per sample
            avg_halls = [comparison['hallucination_comparison'][m]['avg_hallucinations_per_sample']
                        for m in models]
            ax2.bar(models, avg_halls, alpha=0.7, color='darkred', edgecolor='black')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Average Hallucinations per Sample')
            ax2.set_title('Average Number of Hallucinations per Sample')

            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison_hallucinations.png', dpi=150)
            plt.close()

    def generate_html_dashboard(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)

        # Generate all visualizations
        self.generate_visualizations(output_dir)

        # Create HTML
        html_content = self._generate_html_content(output_dir)

        html_path = output_dir / 'dashboard.html'
        with open(html_path, 'w') as f:
            f.write(html_content)

        return html_path

    def _generate_html_content(self, output_dir: Path) -> str:
        # Analyze data
        single_analyses = {model: self.analyze_single_file(model)
                          for model in self.results_data}
        comparison = self.compare_models() if len(self.results_data) > 1 else None

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Label Results Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
        }
        .tab button:hover {
            background-color: #ddd;
        }
        .tab button.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Label Results Analysis Dashboard</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

        <div class="tab">
"""

        # Add tabs for each model
        for i, model in enumerate(self.results_data):
            active = "active" if i == 0 else ""
            html += f'            <button class="tablinks {active}" onclick="openTab(event, \'{model}\')">{model}</button>\n'

        if comparison:
            html += '            <button class="tablinks" onclick="openTab(event, \'comparison\')">Model Comparison</button>\n'

        html += """        </div>
"""

        # Add content for each model
        for i, (model, analysis) in enumerate(single_analyses.items()):
            display = "block" if i == 0 else "none"
            html += f"""
        <div id="{model}" class="tabcontent" style="display: {display};">
            <h2>{model} Analysis</h2>

            <h3>Overall Metrics</h3>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{analysis['overall_metrics']['precision']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{analysis['overall_metrics']['recall']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{analysis['overall_metrics']['f1_score']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{analysis['overall_metrics']['accuracy']:.3f}</div>
                </div>
            </div>

            <h3>Dataset Statistics</h3>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{analysis['total_samples']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Categories</div>
                    <div class="metric-value">{analysis['total_categories']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Predictions/Image</div>
                    <div class="metric-value">{analysis['response_length_stats']['mean_predictions']:.1f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Ground Truth/Image</div>
                    <div class="metric-value">{analysis['response_length_stats']['mean_ground_truth']:.1f}</div>
                </div>
            </div>

            <h3>Hallucination Statistics</h3>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Hallucinated Categories</div>
                    <div class="metric-value">{analysis['hallucination_stats']['total_hallucinated_categories']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Samples with Hallucinations</div>
                    <div class="metric-value">{analysis['hallucination_stats']['samples_with_hallucinations']} ({analysis['hallucination_stats']['hallucination_rate']*100:.1f}%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Hallucinations/Sample</div>
                    <div class="metric-value">{analysis['hallucination_stats']['avg_hallucinations_per_sample']:.2f}</div>
                </div>
            </div>

            <h3>Visualizations</h3>
            <div class="image-grid">
                <div>
                    <img src="{model}_category_performance.png" alt="Category Performance">
                </div>
                <div>
                    <img src="{model}_metric_distributions.png" alt="Metric Distributions">
                </div>
                <div>
                    <img src="{model}_error_analysis.png" alt="Error Analysis">
                </div>
"""
            # Add hallucination visualization if it exists
            if analysis['hallucination_stats']['total_hallucinated_categories'] > 0:
                html += f"""                <div>
                    <img src="{model}_hallucination_analysis.png" alt="Hallucination Analysis">
                </div>
"""

            html += """            </div>

            <h3>Top Performing Categories</h3>
            <table>
                <tr><th>Category</th><th>F1 Score</th></tr>
"""
            for cat, score in analysis['top_10_categories']:
                html += f"                <tr><td>{cat}</td><td>{score:.3f}</td></tr>\n"

            html += """            </table>

            <h3>Bottom Performing Categories</h3>
            <table>
                <tr><th>Category</th><th>F1 Score</th></tr>
"""
            for cat, score in analysis['bottom_10_categories']:
                html += f"                <tr><td>{cat}</td><td>{score:.3f}</td></tr>\n"

            html += """            </table>
"""

            # Add hallucinated categories if any
            if analysis['hallucination_stats']['total_hallucinated_categories'] > 0:
                html += """
            <h3>Hallucinated Categories (Not in Ground Truth)</h3>
            <table>
                <tr><th>Category</th><th>False Positive Count</th></tr>
"""
                hall_cats = analysis['hallucination_stats']['hallucinated_category_list']
                hall_counts = analysis['hallucination_stats']['hallucination_fp_counts']
                for cat in sorted(hall_cats, key=lambda x: hall_counts.get(x, 0), reverse=True)[:10]:
                    count = hall_counts.get(cat, 0)
                    html += f"                <tr><td>{cat}</td><td>{count}</td></tr>\n"

                html += """            </table>
"""

            html += """        </div>
"""

        # Add comparison tab if multiple models
        if comparison:
            html += """
        <div id="comparison" class="tabcontent">
            <h2>Model Comparison</h2>

            <h3>Overall Performance</h3>
            <img src="model_comparison_overall.png" alt="Overall Performance Comparison">

            <h3>Processing Speed</h3>
            <img src="model_comparison_speed.png" alt="Processing Speed Comparison">

            <h3>Category Performance Differences</h3>
            <img src="model_comparison_category_differences.png" alt="Category Differences">

            <h3>Hallucination Comparison</h3>
            <img src="model_comparison_hallucinations.png" alt="Hallucination Comparison">

            <h3>Detailed Metrics Comparison</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Accuracy</th>
                </tr>
"""
            for model in comparison['models']:
                metrics = comparison['overall_comparison'][model]
                html += f"""                <tr>
                    <td>{model}</td>
                    <td>{metrics['precision']:.3f}</td>
                    <td>{metrics['recall']:.3f}</td>
                    <td>{metrics['f1_score']:.3f}</td>
                    <td>{metrics['accuracy']:.3f}</td>
                </tr>
"""

            html += """            </table>
        </div>
"""

        html += """
    </div>

    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }
    </script>
</body>
</html>
"""

        return html


def main():
    parser = argparse.ArgumentParser(description="Analyze label evaluation results")
    parser.add_argument("json_files", nargs="+",
                       help="Path(s) to JSON result files from evaluate_labels_only.py")
    parser.add_argument("--output-dir", default="analysis_dashboard",
                       help="Output directory for analysis dashboard (default: analysis_dashboard)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Validate input files
    for file_path in args.json_files:
        if not Path(file_path).exists():
            logging.error(f"File not found: {file_path}")
            return

    # Create analyzer
    analyzer = LabelResultsAnalyzer(args.json_files)

    # Generate dashboard
    output_dir = Path(args.output_dir)
    logging.info(f"Generating dashboard in {output_dir}")

    html_path = analyzer.generate_html_dashboard(output_dir)

    print(f"\nDashboard generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Dashboard HTML: {html_path}")

    # Open in browser
    if not args.no_browser:
        webbrowser.open(f"file://{html_path.absolute()}")
        print("Dashboard opened in browser")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for model_name in analyzer.results_data:
        analysis = analyzer.analyze_single_file(model_name)
        print(f"\n{model_name}:")
        print(f"  Overall F1: {analysis['overall_metrics']['f1_score']:.3f}")
        print(f"  Precision: {analysis['overall_metrics']['precision']:.3f}")
        print(f"  Recall: {analysis['overall_metrics']['recall']:.3f}")
        print(f"  Accuracy: {analysis['overall_metrics']['accuracy']:.3f}")
        print(f"  Total samples: {analysis['total_samples']}")

        # Hallucination stats
        hall_stats = analysis['hallucination_stats']
        print(f"\n  Hallucination Statistics:")
        print(f"    Hallucinated categories: {hall_stats['total_hallucinated_categories']}")
        print(f"    Samples with hallucinations: {hall_stats['samples_with_hallucinations']} ({hall_stats['hallucination_rate']*100:.1f}%)")
        print(f"    Avg hallucinations per sample: {hall_stats['avg_hallucinations_per_sample']:.2f}")

        if hall_stats['hallucinated_category_list']:
            print(f"    Top hallucinated categories: {', '.join(hall_stats['hallucinated_category_list'][:5])}")


if __name__ == "__main__":
    main()

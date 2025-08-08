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


class BBoxResultsAnalyzer:
    def __init__(self, json_files: List[str]):
        self.results_data = {}
        self.load_results(json_files)

    def load_results(self, json_files: List[str]):
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract model name from path
                model_name = Path(file_path).stem
                self.results_data[model_name] = data

    def analyze_single_file(self, model_name: str) -> Dict:
        data = self.results_data[model_name]

        # Extract core metrics
        coco_metrics = data['coco_evaluation']['metrics']
        location_stats = data['location_statistics']

        # Extract per-category AP
        per_category_ap = coco_metrics.get('per_category_AP', {})
        
        # hack to filter out hallucinations
        per_category_ap = {k: v for k, v in per_category_ap.items() if v != 0.0}

        # Sort categories by AP
        category_ap_sorted = sorted(per_category_ap.items(), key=lambda x: x[1], reverse=True)

        # Get top and bottom performing categories
        top_10_categories = category_ap_sorted[:10]
        bottom_10_categories = category_ap_sorted[-10:] if len(category_ap_sorted) > 10 else []

        # Processing time statistics
        processing_stats = {
            'avg_time_per_image': location_stats['average_processing_time_per_image'],
            'total_images': location_stats['total_images_processed'],
            'total_predictions': location_stats['total_items_predicted'],
            'total_detections': location_stats['total_successful_detections']
        }

        # IoU threshold metrics
        iou_metrics = {
            'AP': coco_metrics['AP'],
            'AP50': coco_metrics['AP50'],
            'AP75': coco_metrics['AP75']
        }

        # Size-based metrics
        size_metrics = {
            'APs': coco_metrics['APs'],  # small objects
            'APm': coco_metrics['APm'],  # medium objects
            'APl': coco_metrics['APl']   # large objects
        }

        # Recall metrics
        recall_metrics = {
            'AR1': coco_metrics['AR1'],    # 1 detection per image
            'AR10': coco_metrics['AR10'],  # 10 detections per image
            'AR100': coco_metrics['AR100'] # 100 detections per image
        }

        return {
            'overall_metrics': coco_metrics,
            'processing_stats': processing_stats,
            'top_10_categories': top_10_categories,
            'bottom_10_categories': bottom_10_categories,
            'iou_metrics': iou_metrics,
            'size_metrics': size_metrics,
            'recall_metrics': recall_metrics,
            'total_categories': len(per_category_ap)
        }

    def compare_models(self) -> Dict:
        if len(self.results_data) < 2:
            return {}

        comparison = {
            'models': list(self.results_data.keys()),
            'overall_comparison': {},
            'category_comparison': defaultdict(dict),
            'processing_comparison': {},
            'iou_comparison': {},
            'size_comparison': {}
        }

        # Overall metrics comparison
        for model_name, data in self.results_data.items():
            metrics = data['coco_evaluation']['metrics']
            stats = data['location_statistics']

            comparison['overall_comparison'][model_name] = {
                'AP': metrics['AP'],
                'AP50': metrics['AP50'],
                'AP75': metrics['AP75']
            }

            comparison['processing_comparison'][model_name] = {
                'avg_time_per_image': stats['average_processing_time_per_image']
            }

            comparison['iou_comparison'][model_name] = {
                'AP': metrics['AP'],
                'AP50': metrics['AP50'],
                'AP75': metrics['AP75']
            }

            comparison['size_comparison'][model_name] = {
                'APs': metrics['APs'],
                'APm': metrics['APm'],
                'APl': metrics['APl']
            }

            # Per-category comparison
            per_category_ap = metrics.get('per_category_AP', {})
            for category, ap in per_category_ap.items():
                comparison['category_comparison'][category][model_name] = ap

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
        analysis = self.analyze_single_file(model_name)

        # 1. Category AP scores (top and bottom)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Top 10 categories
        if analysis['top_10_categories']:
            top_cats = [cat for cat, _ in analysis['top_10_categories']]
            top_scores = [score for _, score in analysis['top_10_categories']]
            ax1.barh(top_cats, top_scores, color='green', alpha=0.7)
            ax1.set_xlabel('Average Precision (AP)')
            ax1.set_title(f'Top 10 Categories by AP - {model_name}')
            ax1.set_xlim(0, 1)

        # Bottom 10 categories
        if analysis['bottom_10_categories']:
            bottom_cats = [cat for cat, _ in analysis['bottom_10_categories']]
            bottom_scores = [score for _, score in analysis['bottom_10_categories']]
            ax2.barh(bottom_cats, bottom_scores, color='red', alpha=0.7)
            ax2.set_xlabel('Average Precision (AP)')
            ax2.set_title(f'Bottom 10 Categories by AP - {model_name}')
            ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_category_ap.png', dpi=150)
        plt.close()

        # 2. IoU threshold comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        iou_metrics = analysis['iou_metrics']
        thresholds = ['AP\n(0.5:0.95)', 'AP50\n(0.5)', 'AP75\n(0.75)']
        values = [iou_metrics['AP'], iou_metrics['AP50'], iou_metrics['AP75']]

        bars = ax.bar(thresholds, values, color=['blue', 'lightblue', 'darkblue'], alpha=0.7)
        ax.set_ylabel('Average Precision')
        ax.set_title(f'Average Precision at Different IoU Thresholds - {model_name}')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_iou_metrics.png', dpi=150)
        plt.close()

        # 3. Size-based AP metrics
        fig, ax = plt.subplots(figsize=(10, 6))

        size_metrics = analysis['size_metrics']
        sizes = ['Small\n(area < 32²)', 'Medium\n(32² < area < 96²)', 'Large\n(area > 96²)']
        values = [size_metrics['APs'], size_metrics['APm'], size_metrics['APl']]

        bars = ax.bar(sizes, values, color=['orange', 'darkorange', 'red'], alpha=0.7)
        ax.set_ylabel('Average Precision')
        ax.set_title(f'Average Precision by Object Size - {model_name}')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_size_metrics.png', dpi=150)
        plt.close()

    def _generate_comparison_plots(self, output_dir: Path):
        comparison = self.compare_models()

        # 1. Overall AP comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(comparison['overall_comparison'].keys())
        metrics = ['AP', 'AP50', 'AP75']
        x = np.arange(len(models))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [comparison['overall_comparison'][m][metric] for m in models]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision')
        ax.set_title('Overall AP Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_ap.png', dpi=150)
        plt.close()

        # 2. Processing speed comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(comparison['processing_comparison'].keys())
        times = [comparison['processing_comparison'][m]['avg_time_per_image'] for m in models]

        bars = ax.bar(models, times, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Time per Image (seconds)')
        ax.set_title('Processing Speed Comparison')

        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_speed.png', dpi=150)
        plt.close()

        # 3. Size-based AP comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(comparison['size_comparison'].keys())
        size_types = ['APs', 'APm', 'APl']
        size_labels = ['Small', 'Medium', 'Large']
        x = np.arange(len(models))
        width = 0.25

        for i, (size_type, label) in enumerate(zip(size_types, size_labels)):
            values = [comparison['size_comparison'][m][size_type] for m in models]
            ax.bar(x + i * width, values, width, label=label, alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision')
        ax.set_title('AP by Object Size Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_size.png', dpi=150)
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
    <title>BBox Results Analysis Dashboard</title>
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
        <h1>BBox Detection Results Analysis Dashboard</h1>
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

            <h3>Overview Metrics</h3>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">COCO AP (IoU 0.5:0.95)</div>
                    <div class="metric-value">{analysis['overall_metrics']['AP']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">AP50 (IoU 0.5)</div>
                    <div class="metric-value">{analysis['overall_metrics']['AP50']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">AP75 (IoU 0.75)</div>
                    <div class="metric-value">{analysis['overall_metrics']['AP75']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Time/Image</div>
                    <div class="metric-value">{analysis['processing_stats']['avg_time_per_image']:.2f}s</div>
                </div>
            </div>

            <h3>Additional Metrics</h3>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Total Categories</div>
                    <div class="metric-value">{analysis['total_categories']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Images</div>
                    <div class="metric-value">{analysis['processing_stats']['total_images']}</div>
                </div>
            </div>

            <h3>Visualizations</h3>
            <div class="image-grid">
                <div>
                    <img src="{model}_category_ap.png" alt="Category AP Performance">
                </div>
                <div>
                    <img src="{model}_iou_metrics.png" alt="IoU Threshold Metrics">
                </div>
                <div>
                    <img src="{model}_size_metrics.png" alt="Size-based Metrics">
                </div>
            </div>

            <h3>Top Performing Categories (by AP)</h3>
            <table>
                <tr><th>Category</th><th>Average Precision</th></tr>
"""
            for cat, score in analysis['top_10_categories']:
                html += f"                <tr><td>{cat}</td><td>{score:.3f}</td></tr>\n"

            html += """            </table>

            <h3>Bottom Performing Categories (by AP)</h3>
            <table>
                <tr><th>Category</th><th>Average Precision</th></tr>
"""
            for cat, score in analysis['bottom_10_categories']:
                html += f"                <tr><td>{cat}</td><td>{score:.3f}</td></tr>\n"

            html += """            </table>

            <h3>IoU Threshold Impact</h3>
            <p>The IoU (Intersection over Union) threshold determines how strict the evaluation is.
            Higher thresholds require more precise bounding box alignment with ground truth.</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>IoU Threshold</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>AP</td>
                    <td>0.50:0.95 (averaged)</td>
                    <td>{:.3f}</td>
                </tr>
                <tr>
                    <td>AP50</td>
                    <td>0.50</td>
                    <td>{:.3f}</td>
                </tr>
                <tr>
                    <td>AP75</td>
                    <td>0.75</td>
                    <td>{:.3f}</td>
                </tr>
            </table>
""".format(
                analysis['iou_metrics']['AP'],
                analysis['iou_metrics']['AP50'],
                analysis['iou_metrics']['AP75']
            )

            html += """        </div>
"""

        # Add comparison tab if multiple models
        if comparison:
            html += """
        <div id="comparison" class="tabcontent">
            <h2>Model Comparison</h2>

            <h3>Overall Performance</h3>
            <img src="model_comparison_ap.png" alt="AP Metrics Comparison">

            <h3>Processing Speed</h3>
            <img src="model_comparison_speed.png" alt="Processing Speed Comparison">

            <h3>Performance by Object Size</h3>
            <img src="model_comparison_size.png" alt="Size-based Performance">

            <h3>Detailed Metrics Comparison</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>AP</th>
                    <th>AP50</th>
                    <th>AP75</th>
                    <th>Avg Time/Image</th>
                </tr>
"""
            for model in comparison['models']:
                ap_metrics = comparison['overall_comparison'][model]
                speed = comparison['processing_comparison'][model]['avg_time_per_image']
                html += f"""                <tr>
                    <td>{model}</td>
                    <td>{ap_metrics['AP']:.3f}</td>
                    <td>{ap_metrics['AP50']:.3f}</td>
                    <td>{ap_metrics['AP75']:.3f}</td>
                    <td>{speed:.2f}s</td>
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
    parser = argparse.ArgumentParser(description="Analyze bounding box detection results")
    parser.add_argument("json_files", nargs="+",
                       help="Path(s) to JSON result files from detect_locations.py")
    parser.add_argument("--output-dir", default="bbox_analysis_dashboard",
                       help="Output directory for analysis dashboard (default: bbox_analysis_dashboard)")
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
    analyzer = BBoxResultsAnalyzer(args.json_files)

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
        print(f"  COCO AP: {analysis['overall_metrics']['AP']:.3f}")
        print(f"  AP50: {analysis['overall_metrics']['AP50']:.3f}")
        print(f"  AP75: {analysis['overall_metrics']['AP75']:.3f}")
        print(f"  Avg Time/Image: {analysis['processing_stats']['avg_time_per_image']:.2f}s")
        print(f"  Total Categories: {analysis['total_categories']}")


if __name__ == "__main__":
    main()

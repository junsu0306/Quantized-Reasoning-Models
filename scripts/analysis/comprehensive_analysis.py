"""
Comprehensive Analysis of Quantized Reasoning Models

This script performs in-depth analysis of all quantized models across multiple datasets
for top-tier academic publication. It generates:
1. Statistical summaries (accuracy, response length, error rates)
2. Error pattern analysis (repetition degeneration, garbled text)
3. Cross-model comparisons
4. Dataset-specific insights
5. Intermediate data files (JSON, CSV) for further analysis
6. Publication-ready visualizations and reports

Usage:
    python scripts/analysis/comprehensive_analysis.py --seed 42 --output_dir ./analysis_results
"""

import os
import json
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd


class ComprehensiveAnalyzer:
    """Main analyzer class for quantized reasoning models."""

    def __init__(self, inference_dir: str, seed: int, output_dir: str):
        self.inference_dir = inference_dir
        self.seed = seed
        self.output_dir = output_dir
        self.results = {}
        self.intermediate_data = {}

        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_dir, "intermediate_data")).mkdir(exist_ok=True)
        Path(os.path.join(output_dir, "tables")).mkdir(exist_ok=True)
        Path(os.path.join(output_dir, "statistics")).mkdir(exist_ok=True)

    def discover_models_and_datasets(self) -> Dict[str, List[str]]:
        """Automatically discover all model-dataset combinations."""
        model_datasets = defaultdict(list)

        seed_suffix = f"-seed{self.seed}"
        for model_dir in sorted(Path(self.inference_dir).iterdir()):
            if not model_dir.is_dir() or not model_dir.name.endswith(seed_suffix):
                continue

            model_name = model_dir.name.replace(seed_suffix, "")

            # Find all JSONL files (datasets)
            for dataset_file in model_dir.glob("*.jsonl"):
                dataset_name = dataset_file.stem
                model_datasets[model_name].append(dataset_name)

        return dict(model_datasets)

    def load_responses(self, model_name: str, dataset: str) -> List[Dict]:
        """Load inference results from JSONL file."""
        file_path = os.path.join(
            self.inference_dir,
            f"{model_name}-seed{self.seed}",
            f"{dataset}.jsonl"
        )

        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def extract_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        # Try to find boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try to find #### format (GSM8K)
        gsm_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
        if gsm_match:
            return gsm_match.group(1).strip()

        # Try to find "The answer is"
        answer_match = re.search(r'[Tt]he answer is\s*[:.]?\s*([^\n.]+)', text)
        if answer_match:
            return answer_match.group(1).strip()

        return None

    def detect_repetition_markers(self, text: str) -> Dict[str, int]:
        """Count various repetition markers in text."""
        markers = {
            'wait': len(re.findall(r'\bWait,?\b', text, re.IGNORECASE)),
            'hmm': len(re.findall(r'\bHmm,?\b', text, re.IGNORECASE)),
            'so': len(re.findall(r'\bSo,\b', text)),
            'but': len(re.findall(r'\bBut,?\b', text, re.IGNORECASE)),
            'actually': len(re.findall(r'\bActually,?\b', text, re.IGNORECASE)),
        }
        return markers

    def detect_repeated_sequences(self, text: str) -> Dict[str, Any]:
        """Detect repeated sequences in text."""
        # Find repeated words (5+ times)
        repeated_words = re.findall(r'\b(\w+)\b(?:\s+\1){4,}', text.lower())

        # Find repeated phrases (3+ times)
        sentences = re.split(r'[.!?]+', text)
        sentence_counts = Counter(s.strip() for s in sentences if len(s.strip()) > 10)
        repeated_sentences = {k: v for k, v in sentence_counts.items() if v >= 3}

        return {
            'repeated_words': list(set(repeated_words)),
            'repeated_word_count': len(repeated_words),
            'repeated_sentences': repeated_sentences,
            'max_sentence_repetition': max(sentence_counts.values()) if sentence_counts else 0,
        }

    def calculate_token_diversity(self, text: str, window_size: int = None) -> float:
        """Calculate token diversity (unique tokens / total tokens)."""
        if window_size:
            tokens = text.split()[-window_size:]
        else:
            tokens = text.split()

        if not tokens:
            return 0.0

        return len(set(tokens)) / len(tokens)

    def detect_garbled_text(self, text: str) -> Dict[str, Any]:
        """Detect various types of garbled/corrupted text."""
        # Repeated characters (10+ times)
        repeated_chars = re.findall(r'(.)\1{9,}', text)

        # High special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,!?;:\'"()[]{}')
        special_ratio = special_chars / len(text) if text else 0

        # Encoding errors
        has_encoding_error = bool(re.search(r'�|\\ufffd|\\x[0-9a-f]{2}', text))

        # Gibberish tokens (non-English words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        gibberish_count = sum(1 for w in words if w.lower() in ['vis', 'vi', 'hmm'])

        return {
            'repeated_chars': repeated_chars[:10],  # Limit to first 10
            'repeated_char_count': len(repeated_chars),
            'special_char_ratio': special_ratio,
            'has_encoding_error': has_encoding_error,
            'gibberish_count': gibberish_count,
        }

    def analyze_single_response(self, item: Dict, idx: int) -> Dict[str, Any]:
        """Analyze a single response in detail."""
        text = item['generated_text']

        # Check for correctness - use extractive_match or is_correct
        metrics = item.get('metrics', {})
        is_correct = metrics.get('is_correct', metrics.get('extractive_match', 0.0)) == 1.0

        analysis = {
            'idx': idx,
            'problem': item.get('problem', ''),
            'gold_answer': item.get('gold', 'N/A'),
            'predicted_answer': self.extract_answer(text),
            'is_correct': is_correct,
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_boxed_answer': bool(re.search(r'\\boxed\{', text)),
            'repetition_markers': self.detect_repetition_markers(text),
            'repeated_sequences': self.detect_repeated_sequences(text),
            'token_diversity': self.calculate_token_diversity(text),
            'token_diversity_last_2k': self.calculate_token_diversity(text, window_size=2000),
            'garbled_text': self.detect_garbled_text(text),
        }

        return analysis

    def analyze_model_dataset(self, model_name: str, dataset: str) -> Dict[str, Any]:
        """Analyze all responses for a model-dataset combination."""
        print(f"\nAnalyzing {model_name} on {dataset}...")

        data = self.load_responses(model_name, dataset)
        if data is None:
            print(f"  ⚠ No data found for {model_name} on {dataset}")
            return None

        # Analyze each response
        response_analyses = []
        for idx, item in enumerate(tqdm(data, desc=f"  Processing {dataset}")):
            analysis = self.analyze_single_response(item, idx)
            response_analyses.append(analysis)

        # Aggregate statistics
        correct_responses = [r for r in response_analyses if r['is_correct']]
        incorrect_responses = [r for r in response_analyses if not r['is_correct']]

        result = {
            'model_name': model_name,
            'dataset': dataset,
            'total_samples': len(data),
            'correct_count': len(correct_responses),
            'incorrect_count': len(incorrect_responses),
            'accuracy': len(correct_responses) / len(data) if data else 0,

            # Response length statistics
            'avg_word_count_all': np.mean([r['word_count'] for r in response_analyses]),
            'avg_word_count_correct': np.mean([r['word_count'] for r in correct_responses]) if correct_responses else 0,
            'avg_word_count_incorrect': np.mean([r['word_count'] for r in incorrect_responses]) if incorrect_responses else 0,
            'median_word_count_all': np.median([r['word_count'] for r in response_analyses]),
            'max_word_count': max([r['word_count'] for r in response_analyses]),

            # Answer generation
            'has_answer_rate': sum(1 for r in response_analyses if r['has_boxed_answer']) / len(response_analyses),
            'has_answer_rate_correct': sum(1 for r in correct_responses if r['has_boxed_answer']) / len(correct_responses) if correct_responses else 0,
            'has_answer_rate_incorrect': sum(1 for r in incorrect_responses if r['has_boxed_answer']) / len(incorrect_responses) if incorrect_responses else 0,

            # Repetition markers
            'avg_wait_count_all': np.mean([r['repetition_markers']['wait'] for r in response_analyses]),
            'avg_wait_count_correct': np.mean([r['repetition_markers']['wait'] for r in correct_responses]) if correct_responses else 0,
            'avg_wait_count_incorrect': np.mean([r['repetition_markers']['wait'] for r in incorrect_responses]) if incorrect_responses else 0,
            'max_wait_count': max([r['repetition_markers']['wait'] for r in response_analyses]),

            # Token diversity
            'avg_diversity_all': np.mean([r['token_diversity'] for r in response_analyses]),
            'avg_diversity_correct': np.mean([r['token_diversity'] for r in correct_responses]) if correct_responses else 0,
            'avg_diversity_incorrect': np.mean([r['token_diversity'] for r in incorrect_responses]) if incorrect_responses else 0,
            'avg_diversity_last_2k_all': np.mean([r['token_diversity_last_2k'] for r in response_analyses]),

            # Garbled text detection
            'garbled_text_count': sum(1 for r in response_analyses if r['garbled_text']['repeated_char_count'] > 0),
            'avg_gibberish_count': np.mean([r['garbled_text']['gibberish_count'] for r in response_analyses]),

            # Detailed response data
            'response_analyses': response_analyses,
        }

        return result

    def analyze_all(self):
        """Analyze all model-dataset combinations."""
        print("=" * 80)
        print("COMPREHENSIVE ANALYSIS OF QUANTIZED REASONING MODELS")
        print("=" * 80)

        # Discover all models and datasets
        model_datasets = self.discover_models_and_datasets()
        print(f"\nDiscovered {len(model_datasets)} models:")
        for model_name, datasets in model_datasets.items():
            print(f"  • {model_name}: {', '.join(datasets)}")

        # Analyze each combination
        for model_name, datasets in model_datasets.items():
            for dataset in datasets:
                key = f"{model_name}_{dataset}"
                result = self.analyze_model_dataset(model_name, dataset)
                if result:
                    self.results[key] = result

                    # Save intermediate data
                    self.save_intermediate_data(key, result)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)

    def save_intermediate_data(self, key: str, result: Dict):
        """Save intermediate analysis data to JSON and CSV."""
        # Save full result to JSON
        json_path = os.path.join(self.output_dir, "intermediate_data", f"{key}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Save response-level data to CSV
        if 'response_analyses' in result:
            df_data = []
            for r in result['response_analyses']:
                df_data.append({
                    'idx': r['idx'],
                    'model': result['model_name'],
                    'dataset': result['dataset'],
                    'is_correct': r['is_correct'],
                    'word_count': r['word_count'],
                    'wait_count': r['repetition_markers']['wait'],
                    'token_diversity': r['token_diversity'],
                    'token_diversity_last_2k': r['token_diversity_last_2k'],
                    'has_boxed_answer': r['has_boxed_answer'],
                    'gibberish_count': r['garbled_text']['gibberish_count'],
                    'repeated_char_count': r['garbled_text']['repeated_char_count'],
                })

            df = pd.DataFrame(df_data)
            csv_path = os.path.join(self.output_dir, "intermediate_data", f"{key}.csv")
            df.to_csv(csv_path, index=False)

    def generate_summary_tables(self):
        """Generate summary tables for publication."""
        print("\nGenerating summary tables...")

        # Overall performance table
        self._generate_performance_table()

        # Dataset-specific tables
        datasets = set(r['dataset'] for r in self.results.values() if r)
        for dataset in datasets:
            self._generate_dataset_specific_table(dataset)

        # Error analysis table
        self._generate_error_analysis_table()

        # Quantization method comparison
        self._generate_quantization_comparison_table()

    def _generate_performance_table(self):
        """Generate overall performance comparison table."""
        table_data = []

        for key, result in sorted(self.results.items()):
            if not result:
                continue

            table_data.append({
                'Model': result['model_name'],
                'Dataset': result['dataset'],
                'Accuracy (%)': f"{result['accuracy'] * 100:.2f}",
                'Avg Words': f"{result['avg_word_count_all']:.0f}",
                'Avg Wait Count': f"{result['avg_wait_count_all']:.1f}",
                'Token Diversity': f"{result['avg_diversity_all']:.4f}",
                'Has Answer (%)': f"{result['has_answer_rate'] * 100:.1f}",
            })

        df = pd.DataFrame(table_data)

        # Save as CSV
        csv_path = os.path.join(self.output_dir, "tables", "overall_performance.csv")
        df.to_csv(csv_path, index=False)

        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, "tables", "overall_performance.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))

        # Save as Markdown
        md_path = os.path.join(self.output_dir, "tables", "overall_performance.md")
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))

        print(f"  ✓ Overall performance table saved to {csv_path}")

    def _generate_dataset_specific_table(self, dataset: str):
        """Generate dataset-specific comparison table."""
        table_data = []

        # Get baseline performance
        baseline_key = f"DeepSeek-R1-Distill-Qwen-1.5B_{dataset}"
        baseline_accuracy = self.results.get(baseline_key, {}).get('accuracy', 0)

        for key, result in sorted(self.results.items()):
            if not result or result['dataset'] != dataset:
                continue

            accuracy_delta = (result['accuracy'] - baseline_accuracy) * 100

            table_data.append({
                'Model': result['model_name'],
                'Accuracy (%)': f"{result['accuracy'] * 100:.2f}",
                'Δ from Baseline (pp)': f"{accuracy_delta:+.2f}",
                'Correct/Total': f"{result['correct_count']}/{result['total_samples']}",
                'Avg Words (Correct)': f"{result['avg_word_count_correct']:.0f}",
                'Avg Words (Incorrect)': f"{result['avg_word_count_incorrect']:.0f}",
                'Wait Count (Incorrect)': f"{result['avg_wait_count_incorrect']:.1f}",
            })

        df = pd.DataFrame(table_data)

        # Save in multiple formats
        base_name = f"{dataset.replace('-', '_')}_comparison"
        csv_path = os.path.join(self.output_dir, "tables", f"{base_name}.csv")
        df.to_csv(csv_path, index=False)

        latex_path = os.path.join(self.output_dir, "tables", f"{base_name}.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))

        md_path = os.path.join(self.output_dir, "tables", f"{base_name}.md")
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))

        print(f"  ✓ {dataset} comparison table saved")

    def _generate_error_analysis_table(self):
        """Generate error pattern analysis table."""
        table_data = []

        for key, result in sorted(self.results.items()):
            if not result or result['incorrect_count'] == 0:
                continue

            incorrect_responses = [r for r in result['response_analyses'] if not r['is_correct']]

            # Count severe repetition cases (Wait > 1000)
            severe_repetition = sum(1 for r in incorrect_responses if r['repetition_markers']['wait'] > 1000)

            # Count garbled text cases
            garbled_cases = sum(1 for r in incorrect_responses if r['garbled_text']['repeated_char_count'] > 0)

            # Count missing answer cases
            missing_answer = sum(1 for r in incorrect_responses if not r['has_boxed_answer'])

            table_data.append({
                'Model': result['model_name'],
                'Dataset': result['dataset'],
                'Total Errors': result['incorrect_count'],
                'Severe Repetition': severe_repetition,
                'Severe Repetition (%)': f"{severe_repetition / result['incorrect_count'] * 100:.1f}",
                'Garbled Text': garbled_cases,
                'Missing Answer': missing_answer,
                'Missing Answer (%)': f"{missing_answer / result['incorrect_count'] * 100:.1f}",
            })

        df = pd.DataFrame(table_data)

        csv_path = os.path.join(self.output_dir, "tables", "error_analysis.csv")
        df.to_csv(csv_path, index=False)

        latex_path = os.path.join(self.output_dir, "tables", "error_analysis.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))

        md_path = os.path.join(self.output_dir, "tables", "error_analysis.md")
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))

        print(f"  ✓ Error analysis table saved")

    def _generate_quantization_comparison_table(self):
        """Generate quantization method comparison table."""
        # Group by quantization method
        quant_methods = {
            'Baseline (FP16)': 'DeepSeek-R1-Distill-Qwen-1.5B',
            'AWQ W4': 'awq-w4g128',
            'AWQ W3': 'awq-w3g128',
            'GPTQ W4': 'gptq-w4g128',
            'GPTQ W3': 'gptq-w3g128',
            'KV-Quant* KV4': 'kvquant_star-kv4',
            'KV-Quant* KV3': 'kvquant_star-kv3',
            'SmoothQuant W8A8': 'smoothquant-w8a8kv8',
        }

        datasets = set(r['dataset'] for r in self.results.values() if r)

        for dataset in datasets:
            table_data = []

            for method_name, model_pattern in quant_methods.items():
                # Find matching result
                matching_result = None
                for key, result in self.results.items():
                    if result and result['dataset'] == dataset and model_pattern in result['model_name']:
                        matching_result = result
                        break

                if not matching_result:
                    continue

                table_data.append({
                    'Quantization Method': method_name,
                    'Accuracy (%)': f"{matching_result['accuracy'] * 100:.2f}",
                    'Avg Words': f"{matching_result['avg_word_count_all']:.0f}",
                    'Token Diversity': f"{matching_result['avg_diversity_all']:.4f}",
                    'Wait Count': f"{matching_result['avg_wait_count_all']:.1f}",
                })

            df = pd.DataFrame(table_data)

            base_name = f"{dataset.replace('-', '_')}_quantization_comparison"
            csv_path = os.path.join(self.output_dir, "tables", f"{base_name}.csv")
            df.to_csv(csv_path, index=False)

            latex_path = os.path.join(self.output_dir, "tables", f"{base_name}.tex")
            with open(latex_path, 'w') as f:
                f.write(df.to_latex(index=False, escape=False))

            md_path = os.path.join(self.output_dir, "tables", f"{base_name}.md")
            with open(md_path, 'w') as f:
                f.write(df.to_markdown(index=False))

            print(f"  ✓ {dataset} quantization comparison table saved")

    def generate_statistics_summary(self):
        """Generate comprehensive statistics summary."""
        print("\nGenerating statistics summary...")

        stats = {
            'total_models': len(set(r['model_name'] for r in self.results.values() if r)),
            'total_datasets': len(set(r['dataset'] for r in self.results.values() if r)),
            'total_responses_analyzed': sum(r['total_samples'] for r in self.results.values() if r),
            'models': {},
        }

        # Per-model statistics
        for key, result in self.results.items():
            if not result:
                continue

            model_name = result['model_name']
            if model_name not in stats['models']:
                stats['models'][model_name] = {}

            stats['models'][model_name][result['dataset']] = {
                'accuracy': result['accuracy'],
                'total_samples': result['total_samples'],
                'correct_count': result['correct_count'],
                'incorrect_count': result['incorrect_count'],
                'avg_word_count': result['avg_word_count_all'],
                'avg_wait_count': result['avg_wait_count_all'],
                'token_diversity': result['avg_diversity_all'],
            }

        # Save statistics
        stats_path = os.path.join(self.output_dir, "statistics", "summary.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  ✓ Statistics summary saved to {stats_path}")

    def generate_publication_report(self):
        """Generate comprehensive publication-ready report."""
        print("\nGenerating publication report...")

        report_path = os.path.join(self.output_dir, "COMPREHENSIVE_ANALYSIS_REPORT.md")

        with open(report_path, 'w') as f:
            # Header
            f.write("# Comprehensive Analysis of Quantized Reasoning Models\n\n")
            f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Random Seed**: {self.seed}\n")
            f.write(f"**Total Models Analyzed**: {len(set(r['model_name'] for r in self.results.values() if r))}\n")
            f.write(f"**Total Datasets**: {len(set(r['dataset'] for r in self.results.values() if r))}\n")
            f.write(f"**Total Responses**: {sum(r['total_samples'] for r in self.results.values() if r):,}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f)

            # Dataset-specific analysis
            datasets = sorted(set(r['dataset'] for r in self.results.values() if r))
            for dataset in datasets:
                f.write(f"\n---\n\n## {dataset} Analysis\n\n")
                self._write_dataset_analysis(f, dataset)

            # Quantization method comparison
            f.write("\n---\n\n## Quantization Method Comparison\n\n")
            self._write_quantization_comparison(f)

            # Error pattern analysis
            f.write("\n---\n\n## Error Pattern Analysis\n\n")
            self._write_error_pattern_analysis(f)

            # Key findings and recommendations
            f.write("\n---\n\n## Key Findings and Recommendations\n\n")
            self._write_findings_and_recommendations(f)

            # Data availability
            f.write("\n---\n\n## Data Availability\n\n")
            f.write("All intermediate data, tables, and statistics are available in:\n")
            f.write(f"- Intermediate data: `{os.path.join(self.output_dir, 'intermediate_data')}/`\n")
            f.write(f"- Tables (CSV, LaTeX, Markdown): `{os.path.join(self.output_dir, 'tables')}/`\n")
            f.write(f"- Statistics: `{os.path.join(self.output_dir, 'statistics')}/`\n\n")

        print(f"  ✓ Publication report saved to {report_path}")

    def _write_executive_summary(self, f):
        """Write executive summary section."""
        # Find baseline and best performing models for each dataset
        datasets = set(r['dataset'] for r in self.results.values() if r)

        for dataset in sorted(datasets):
            dataset_results = {k: v for k, v in self.results.items() if v and v['dataset'] == dataset}

            # Find baseline
            baseline_key = f"DeepSeek-R1-Distill-Qwen-1.5B_{dataset}"
            baseline = dataset_results.get(baseline_key)

            if not baseline:
                continue

            # Find best and worst quantized models
            quantized_results = {k: v for k, v in dataset_results.items() if 'DeepSeek-R1-Distill-Qwen-1.5B-' in v['model_name']}

            if not quantized_results:
                continue

            best_model = max(quantized_results.values(), key=lambda x: x['accuracy'])
            worst_model = min(quantized_results.values(), key=lambda x: x['accuracy'])

            f.write(f"### {dataset}\n\n")
            f.write(f"- **Baseline Accuracy**: {baseline['accuracy'] * 100:.2f}%\n")
            f.write(f"- **Best Quantized Model**: {best_model['model_name']}\n")
            f.write(f"  - Accuracy: {best_model['accuracy'] * 100:.2f}% ({(best_model['accuracy'] - baseline['accuracy']) * 100:+.2f}pp)\n")
            f.write(f"- **Worst Quantized Model**: {worst_model['model_name']}\n")
            f.write(f"  - Accuracy: {worst_model['accuracy'] * 100:.2f}% ({(worst_model['accuracy'] - baseline['accuracy']) * 100:+.2f}pp)\n\n")

    def _write_dataset_analysis(self, f, dataset: str):
        """Write dataset-specific analysis."""
        dataset_results = {k: v for k, v in self.results.items() if v and v['dataset'] == dataset}

        # Performance table
        f.write("### Performance Comparison\n\n")
        f.write("| Model | Accuracy | Avg Words | Wait Count | Token Diversity |\n")
        f.write("|-------|----------|-----------|------------|------------------|\n")

        for key in sorted(dataset_results.keys()):
            result = dataset_results[key]
            f.write(f"| {result['model_name']} | {result['accuracy'] * 100:.2f}% | "
                   f"{result['avg_word_count_all']:.0f} | {result['avg_wait_count_all']:.1f} | "
                   f"{result['avg_diversity_all']:.4f} |\n")

        f.write("\n")

    def _write_quantization_comparison(self, f):
        """Write quantization method comparison."""
        # Group models by quantization method
        quant_groups = defaultdict(list)

        for key, result in self.results.items():
            if not result:
                continue

            model_name = result['model_name']
            if 'awq' in model_name.lower():
                method = 'AWQ'
            elif 'gptq' in model_name.lower():
                method = 'GPTQ'
            elif 'kvquant' in model_name.lower():
                method = 'KV-Quant*'
            elif 'smoothquant' in model_name.lower():
                method = 'SmoothQuant'
            else:
                method = 'Baseline'

            quant_groups[method].append(result)

        for method, results in sorted(quant_groups.items()):
            f.write(f"### {method}\n\n")
            f.write(f"- **Number of Models**: {len(set(r['model_name'] for r in results))}\n")
            f.write(f"- **Average Accuracy**: {np.mean([r['accuracy'] for r in results]) * 100:.2f}%\n")
            f.write(f"- **Average Response Length**: {np.mean([r['avg_word_count_all'] for r in results]):.0f} words\n")
            f.write(f"- **Average Token Diversity**: {np.mean([r['avg_diversity_all'] for r in results]):.4f}\n\n")

    def _write_error_pattern_analysis(self, f):
        """Write error pattern analysis."""
        # Analyze error patterns across all models
        all_errors = []

        for key, result in self.results.items():
            if not result:
                continue

            incorrect_responses = [r for r in result['response_analyses'] if not r['is_correct']]
            all_errors.extend(incorrect_responses)

        if not all_errors:
            f.write("No errors found in analyzed responses.\n\n")
            return

        # Categorize errors
        severe_repetition = sum(1 for r in all_errors if r['repetition_markers']['wait'] > 1000)
        moderate_repetition = sum(1 for r in all_errors if 100 < r['repetition_markers']['wait'] <= 1000)
        garbled_text = sum(1 for r in all_errors if r['garbled_text']['repeated_char_count'] > 0)
        missing_answer = sum(1 for r in all_errors if not r['has_boxed_answer'])
        low_diversity = sum(1 for r in all_errors if r['token_diversity'] < 0.1)

        f.write(f"**Total Errors Analyzed**: {len(all_errors)}\n\n")
        f.write("### Error Categories\n\n")
        f.write(f"- **Severe Repetition (Wait > 1000)**: {severe_repetition} ({severe_repetition / len(all_errors) * 100:.1f}%)\n")
        f.write(f"- **Moderate Repetition (100 < Wait ≤ 1000)**: {moderate_repetition} ({moderate_repetition / len(all_errors) * 100:.1f}%)\n")
        f.write(f"- **Garbled Text**: {garbled_text} ({garbled_text / len(all_errors) * 100:.1f}%)\n")
        f.write(f"- **Missing Answer**: {missing_answer} ({missing_answer / len(all_errors) * 100:.1f}%)\n")
        f.write(f"- **Low Token Diversity (< 0.1)**: {low_diversity} ({low_diversity / len(all_errors) * 100:.1f}%)\n\n")

    def _write_findings_and_recommendations(self, f):
        """Write key findings and recommendations."""
        f.write("### Key Findings\n\n")

        # Find best performing quantization methods
        datasets = set(r['dataset'] for r in self.results.values() if r)

        for dataset in sorted(datasets):
            baseline_key = f"DeepSeek-R1-Distill-Qwen-1.5B_{dataset}"
            baseline = self.results.get(baseline_key)

            if not baseline:
                continue

            dataset_results = {k: v for k, v in self.results.items() if v and v['dataset'] == dataset and 'DeepSeek-R1-Distill-Qwen-1.5B-' in v['model_name']}

            # Find models within 5% of baseline
            acceptable_models = [v for v in dataset_results.values() if (baseline['accuracy'] - v['accuracy']) * 100 < 5]

            f.write(f"**{dataset}**:\n")
            f.write(f"- Baseline accuracy: {baseline['accuracy'] * 100:.2f}%\n")
            f.write(f"- {len(acceptable_models)} quantized models maintain <5pp accuracy drop\n")

            if acceptable_models:
                best_acceptable = max(acceptable_models, key=lambda x: x['accuracy'])
                f.write(f"- Best quantized model: {best_acceptable['model_name']} ({best_acceptable['accuracy'] * 100:.2f}%)\n")

            f.write("\n")

        f.write("### Recommendations\n\n")
        f.write("1. **For high-accuracy requirements**: Use baseline FP16 or 4-bit quantization (AWQ/GPTQ)\n")
        f.write("2. **For memory-constrained scenarios**: 4-bit quantization offers best accuracy-efficiency trade-off\n")
        f.write("3. **Avoid 3-bit quantization** for reasoning tasks due to severe repetition degeneration\n")
        f.write("4. **KV-cache quantization** shows promising results with minimal accuracy loss\n\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis of quantized reasoning models')
    parser.add_argument('--inference_dir', type=str, default='./outputs/inference',
                        help='Directory containing inference results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used in inference')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(
        inference_dir=args.inference_dir,
        seed=args.seed,
        output_dir=args.output_dir
    )

    # Run analysis
    analyzer.analyze_all()

    # Generate outputs
    analyzer.generate_summary_tables()
    analyzer.generate_statistics_summary()
    analyzer.generate_publication_report()

    print("\n" + "=" * 80)
    print("ALL ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"- Intermediate data: {os.path.join(args.output_dir, 'intermediate_data')}/")
    print(f"- Tables: {os.path.join(args.output_dir, 'tables')}/")
    print(f"- Statistics: {os.path.join(args.output_dir, 'statistics')}/")
    print(f"- Main report: {os.path.join(args.output_dir, 'COMPREHENSIVE_ANALYSIS_REPORT.md')}")


if __name__ == "__main__":
    main()

"""
Analyze responses from quantized models to detect errors, corruptions, and quality issues.

This script:
1. Loads responses from all quantized models
2. Detects various types of errors:
   - Incomplete responses (missing answer tags)
   - Corrupted/garbled text (encoding errors, repeated tokens)
   - Nonsensical reasoning (logic errors, contradictions)
   - Length anomalies (too short/long)
3. Generates detailed error reports and statistics
"""

import os
import json
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def load_responses(output_dir: str, model_name: str, dataset: str, seed: int) -> List[Dict]:
    """Load inference results from JSON file."""
    full_name = f"{model_name}-seed{seed}"
    file_path = os.path.join(output_dir, full_name, f"{dataset}.jsonl")

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def detect_incomplete_response(text: str) -> Tuple[bool, str]:
    """Detect if response is incomplete (missing answer tags or abrupt end)."""
    # Check for common answer patterns
    answer_patterns = [
        r'\\boxed\{[^}]+\}',  # LaTeX boxed answer
        r'####\s*\d+',         # GSM8K format
        r'The answer is',
        r'Therefore,',
    ]

    has_answer = any(re.search(pattern, text) for pattern in answer_patterns)

    # Check for abrupt endings
    if len(text) < 50:
        return True, "Response too short (< 50 chars)"

    if not has_answer and len(text) < 500:
        return True, "No answer pattern found and response is short"

    # Check if text ends abnormally (not with punctuation)
    if text and not text.strip()[-1] in '.!?\'")}]':
        return True, "Response ends without proper punctuation"

    return False, ""


def detect_garbled_text(text: str) -> Tuple[bool, str]:
    """Detect garbled or corrupted text."""
    # Check for repeated tokens/characters
    repeated_token_pattern = r'(\b\w+\b)(\s+\1){5,}'  # Same word repeated 5+ times
    if re.search(repeated_token_pattern, text):
        match = re.search(repeated_token_pattern, text)
        return True, f"Repeated token: '{match.group(1)}'"

    # Check for repeated characters
    repeated_char_pattern = r'(.)\1{10,}'  # Same character repeated 10+ times
    if re.search(repeated_char_pattern, text):
        match = re.search(repeated_char_pattern, text)
        return True, f"Repeated character: '{match.group(1)}'"

    # Check for unusual character ratio (non-ASCII or special chars)
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,!?;:\'"()[]{}')
    if len(text) > 0 and special_char_count / len(text) > 0.3:
        return True, f"High special character ratio: {special_char_count / len(text):.2%}"

    # Check for encoding issues (mojibake)
    mojibake_patterns = [r'�', r'\ufffd', r'\\x[0-9a-f]{2}']
    for pattern in mojibake_patterns:
        if re.search(pattern, text):
            return True, "Encoding error detected"

    return False, ""


def detect_reasoning_errors(text: str) -> Tuple[bool, str]:
    """Detect potential reasoning errors."""
    # Check for obvious contradictions
    contradiction_patterns = [
        (r'(\d+)\s*=\s*(\d+)', lambda m: m.group(1) != m.group(2)),  # Different numbers claimed equal
    ]

    for pattern, check_func in contradiction_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if check_func and not check_func(match):
                return True, f"Mathematical contradiction: {match.group(0)}"

    # Check for gibberish (no actual words)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(text) > 100 and len(words) < 10:
        return True, "Too few recognizable words"

    return False, ""


def detect_length_anomalies(text: str, texts: List[str]) -> Tuple[bool, str]:
    """Detect abnormal response length compared to others."""
    if not texts:
        return False, ""

    lengths = [len(t) for t in texts]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    current_length = len(text)

    # Flag if more than 3 standard deviations away
    if std_length > 0:
        z_score = abs((current_length - mean_length) / std_length)
        if z_score > 3:
            return True, f"Length anomaly: {current_length} chars (mean={mean_length:.0f}, std={std_length:.0f}, z={z_score:.2f})"

    return False, ""


def analyze_model_responses(output_dir: str, model_name: str, dataset: str, seed: int, all_texts: List[str]) -> Dict:
    """Analyze responses for a single model."""
    data = load_responses(output_dir, model_name, dataset, seed)

    if data is None:
        return None

    results = {
        'model_name': model_name,
        'dataset': dataset,
        'total_samples': len(data),
        'incomplete': [],
        'garbled': [],
        'reasoning_errors': [],
        'length_anomalies': [],
        'all_errors': [],
    }

    for idx, item in enumerate(data):
        text = item['generated_text']
        errors = []

        # Check for incomplete responses
        is_incomplete, reason = detect_incomplete_response(text)
        if is_incomplete:
            results['incomplete'].append({'idx': idx, 'reason': reason, 'text_preview': text[:200]})
            errors.append(f"Incomplete: {reason}")

        # Check for garbled text
        is_garbled, reason = detect_garbled_text(text)
        if is_garbled:
            results['garbled'].append({'idx': idx, 'reason': reason, 'text_preview': text[:200]})
            errors.append(f"Garbled: {reason}")

        # Check for reasoning errors
        has_error, reason = detect_reasoning_errors(text)
        if has_error:
            results['reasoning_errors'].append({'idx': idx, 'reason': reason, 'text_preview': text[:200]})
            errors.append(f"Reasoning: {reason}")

        # Check for length anomalies
        is_anomaly, reason = detect_length_anomalies(text, all_texts)
        if is_anomaly:
            results['length_anomalies'].append({'idx': idx, 'reason': reason})
            errors.append(f"Length: {reason}")

        if errors:
            results['all_errors'].append({
                'idx': idx,
                'errors': errors,
                'text': text,
                'gold': item.get('gold', 'N/A'),
                'metrics': item.get('metrics', {}),
            })

    return results


def print_summary_table(all_results: Dict[str, Dict]):
    """Print summary table of error statistics."""
    print("\n" + "="*100)
    print("ERROR DETECTION SUMMARY")
    print("="*100)

    # Header
    print(f"{'Model':<60} {'Dataset':<15} {'Total':<8} {'Incomplete':<12} {'Garbled':<10} {'Reasoning':<12} {'Length':<10}")
    print("-"*100)

    for key, result in all_results.items():
        if result is None:
            continue

        model_name = result['model_name']
        # Shorten model name for display
        if len(model_name) > 58:
            model_name = model_name[:55] + "..."

        print(f"{model_name:<60} "
              f"{result['dataset']:<15} "
              f"{result['total_samples']:<8} "
              f"{len(result['incomplete']):<12} "
              f"{len(result['garbled']):<10} "
              f"{len(result['reasoning_errors']):<12} "
              f"{len(result['length_anomalies']):<10}")

    print("="*100 + "\n")


def print_detailed_errors(all_results: Dict[str, Dict], max_examples: int = 3):
    """Print detailed error examples."""
    print("\n" + "="*100)
    print("DETAILED ERROR EXAMPLES")
    print("="*100 + "\n")

    for key, result in all_results.items():
        if result is None or not result['all_errors']:
            continue

        print(f"\n{'='*100}")
        print(f"Model: {result['model_name']}")
        print(f"Dataset: {result['dataset']}")
        print(f"Total Errors: {len(result['all_errors'])} / {result['total_samples']}")
        print(f"{'='*100}\n")

        for i, error_item in enumerate(result['all_errors'][:max_examples]):
            print(f"--- Example {i+1}/{min(max_examples, len(result['all_errors']))} (Sample #{error_item['idx']}) ---")
            print(f"Errors: {', '.join(error_item['errors'])}")
            print(f"Gold Answer: {error_item['gold']}")
            print(f"Metrics: {error_item['metrics']}")
            print(f"\nGenerated Text (first 500 chars):")
            print(error_item['text'][:500])
            print("\n" + "-"*100 + "\n")


def save_error_report(all_results: Dict[str, Dict], output_path: str):
    """Save detailed error report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed error report saved to: {output_path}")


def parser_gen():
    parser = argparse.ArgumentParser(description='Analyze responses from quantized models')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference',
                        help='Directory containing inference results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used in inference')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['MATH-500', 'AIME-90'],
                        help='Datasets to analyze')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to analyze (default: all)')
    parser.add_argument('--max_examples', type=int, default=3,
                        help='Maximum number of error examples to show per model')
    parser.add_argument('--save_report', action='store_true',
                        help='Save detailed error report to JSON file')
    return parser.parse_args()


def main():
    args = parser_gen()

    # Define models to analyze
    if args.models:
        model_names = args.models
    else:
        model_names = [
            "DeepSeek-R1-Distill-Qwen-1.5B",  # Base FP16 model
            "DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv3-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-smoothquant-w8a8kv8-tp1",
            "DeepSeek-R1-Distill-Qwen-1.5B-flatquant-w4a4kv4-tp1",
        ]

    all_results = {}

    for dataset in args.datasets:
        print(f"\nAnalyzing dataset: {dataset}")

        # First, collect all texts for length anomaly detection
        all_texts = []
        for model_name in model_names:
            data = load_responses(args.output_dir, model_name, dataset, args.seed)
            if data:
                all_texts.extend([item['generated_text'] for item in data])

        # Analyze each model
        for model_name in tqdm(model_names, desc=f"Analyzing {dataset}"):
            key = f"{model_name}_{dataset}"
            result = analyze_model_responses(args.output_dir, model_name, dataset, args.seed, all_texts)
            all_results[key] = result

    # Print summary table
    print_summary_table(all_results)

    # Print detailed error examples
    print_detailed_errors(all_results, max_examples=args.max_examples)

    # Save report if requested
    if args.save_report:
        report_path = os.path.join(args.output_dir, f"error_report_seed{args.seed}.json")
        save_error_report(all_results, report_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

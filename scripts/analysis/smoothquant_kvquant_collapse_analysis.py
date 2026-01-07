#!/usr/bin/env python3
"""
SmoothQuant 및 KVQuant 모델의 붕괴 현상 분석 스크립트

이 스크립트는 다음 양자화 방법들의 붕괴 현상을 분석합니다:
- SmoothQuant (W8A8KV8)
- KVQuant* (KV3, KV4)
- 비교 대상: Baseline (FP16), AWQ 3-bit, GPTQ 3-bit
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import argparse


def load_inference_results(filepath: str) -> List[Dict]:
    """추론 결과 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answer(text: str) -> Optional[str]:
    """응답에서 \boxed{} 형식의 답변 추출"""
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    return match.group(1) if match else None


def token_diversity(text: str) -> float:
    """토큰 다양성 계산 (고유 토큰 비율)"""
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def count_patterns(text: str) -> Dict[str, int]:
    """반복 마커 패턴 카운트"""
    return {
        'wait': len(re.findall(r'Wait', text, re.IGNORECASE)),
        'perhaps': len(re.findall(r'Perhaps', text, re.IGNORECASE)),
        'let_me': len(re.findall(r'Let me', text, re.IGNORECASE)),
        'hmm': len(re.findall(r'\bHmm\b', text, re.IGNORECASE)),
        'so': len(re.findall(r'\bSo\b', text)),
        'but': len(re.findall(r'\bBut\b', text)),
    }


def classify_collapse_type(text: str, diversity: float, patterns: Dict[str, int]) -> str:
    """붕괴 유형 분류"""
    words = text.split()

    # 토큰 다양성 기반 붕괴 판정
    if diversity < 0.05:
        return "severe_collapse"  # 심각한 붕괴
    elif diversity < 0.10:
        return "moderate_collapse"  # 중간 붕괴
    elif diversity < 0.15 and patterns['wait'] > 100:
        return "wait_loop"  # Wait 반복 루프
    elif patterns['wait'] > 200:
        return "wait_loop"
    elif len(words) > 10000 and diversity < 0.20:
        return "mild_collapse"  # 경미한 붕괴
    else:
        return "no_collapse"  # 붕괴 없음


def analyze_model_results(results: List[Dict], model_name: str) -> Dict:
    """단일 모델의 결과 분석"""
    analysis = {
        'model_name': model_name,
        'total_problems': len(results),
        'collapse_counts': defaultdict(int),
        'answer_generated': 0,
        'total_words': 0,
        'total_wait': 0,
        'total_perhaps': 0,
        'diversity_sum': 0.0,
        'collapse_examples': [],
        'problems_by_type': defaultdict(lambda: {'total': 0, 'collapsed': 0}),
    }

    for i, result in enumerate(results):
        text = result.get('generated_text', '')
        prompt = result.get('full_prompt', '')

        # 기본 통계
        words = len(text.split())
        diversity = token_diversity(text)
        patterns = count_patterns(text)
        answer = extract_answer(text)

        analysis['total_words'] += words
        analysis['total_wait'] += patterns['wait']
        analysis['total_perhaps'] += patterns['perhaps']
        analysis['diversity_sum'] += diversity

        if answer:
            analysis['answer_generated'] += 1

        # 붕괴 유형 분류
        collapse_type = classify_collapse_type(text, diversity, patterns)
        analysis['collapse_counts'][collapse_type] += 1

        # 붕괴 예시 저장 (처음 5개)
        if collapse_type != 'no_collapse' and len(analysis['collapse_examples']) < 5:
            problem_preview = prompt.split('<|User|>')[-1].split('Please reason')[0].strip()[:100]
            analysis['collapse_examples'].append({
                'index': i,
                'problem': problem_preview,
                'collapse_type': collapse_type,
                'words': words,
                'diversity': round(diversity, 4),
                'wait_count': patterns['wait'],
                'answer': answer,
                'text_end': text[-500:] if len(text) > 500 else text
            })

        # 문제 유형별 분석
        problem_type = categorize_problem(prompt)
        analysis['problems_by_type'][problem_type]['total'] += 1
        if collapse_type != 'no_collapse':
            analysis['problems_by_type'][problem_type]['collapsed'] += 1

    # 평균 계산
    n = analysis['total_problems']
    analysis['avg_words'] = analysis['total_words'] / n if n > 0 else 0
    analysis['avg_wait'] = analysis['total_wait'] / n if n > 0 else 0
    analysis['avg_perhaps'] = analysis['total_perhaps'] / n if n > 0 else 0
    analysis['avg_diversity'] = analysis['diversity_sum'] / n if n > 0 else 0
    analysis['answer_rate'] = analysis['answer_generated'] / n * 100 if n > 0 else 0

    # 붕괴율 계산
    collapsed = sum(v for k, v in analysis['collapse_counts'].items() if k != 'no_collapse')
    analysis['collapse_rate'] = collapsed / n * 100 if n > 0 else 0

    return analysis


def categorize_problem(prompt: str) -> str:
    """문제 유형 분류"""
    prompt_lower = prompt.lower()

    if any(w in prompt_lower for w in ['sin', 'cos', 'tan', 'angle', 'degree']):
        return 'trigonometry'
    elif any(w in prompt_lower for w in ['probability', 'chance', 'likely', 'random']):
        return 'probability'
    elif any(w in prompt_lower for w in ['prime', 'divisor', 'factor', 'gcd', 'mod']):
        return 'number_theory'
    elif any(w in prompt_lower for w in ['combination', 'permutation', 'arrange', 'ways', 'choose']):
        return 'combinatorics'
    elif any(w in prompt_lower for w in ['sequence', 'series', 'sum of']):
        return 'sequences'
    elif any(w in prompt_lower for w in ['geometry', 'area', 'perimeter', 'circle', 'sphere', 'triangle']):
        return 'geometry'
    else:
        return 'algebra'


def compare_with_baseline(model_results: List[Dict], baseline_results: List[Dict]) -> Dict:
    """베이스라인과 비교하여 성능 저하 케이스 분석"""
    comparison = {
        'degraded_cases': 0,  # 베이스라인은 정답, 모델은 오답
        'improved_cases': 0,  # 베이스라인은 오답, 모델은 정답
        'both_correct': 0,
        'both_incorrect': 0,
        'model_collapse_baseline_correct': 0,  # 모델 붕괴 + 베이스라인 정답
    }

    for model_res, baseline_res in zip(model_results, baseline_results):
        model_text = model_res.get('generated_text', '')
        baseline_text = baseline_res.get('generated_text', '')

        model_answer = extract_answer(model_text)
        baseline_answer = extract_answer(baseline_text)

        model_diversity = token_diversity(model_text)
        model_collapsed = model_diversity < 0.15 or model_answer is None

        if baseline_answer and model_answer:
            if model_answer == baseline_answer:
                comparison['both_correct'] += 1
            else:
                comparison['degraded_cases'] += 1
        elif baseline_answer and not model_answer:
            comparison['degraded_cases'] += 1
            if model_collapsed:
                comparison['model_collapse_baseline_correct'] += 1
        elif not baseline_answer and model_answer:
            comparison['improved_cases'] += 1
        else:
            comparison['both_incorrect'] += 1

    return comparison


def generate_report(all_analyses: Dict[str, Dict], output_dir: str):
    """분석 보고서 생성"""
    report_path = os.path.join(output_dir, 'SMOOTHQUANT_KVQUANT_COLLAPSE_REPORT.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# SmoothQuant 및 KVQuant 붕괴 현상 분석 보고서\n\n")
        f.write("## Collapse Analysis Report for SmoothQuant and KVQuant Models\n\n")
        f.write("**분석 대상 모델**: DeepSeek-R1-Distill-Qwen-1.5B\n")
        f.write("**데이터셋**: MATH-500, AIME-90\n")
        f.write("**분석 날짜**: 2026-01-07\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## 1. Executive Summary\n\n")
        f.write("### 1.1 주요 발견\n\n")

        # 붕괴율 비교 테이블
        f.write("| 모델 | 데이터셋 | 붕괴율 | 답변 생성률 | 평균 단어 수 | 평균 Wait | 평균 다양성 |\n")
        f.write("|------|---------|--------|-----------|------------|----------|----------|\n")

        for key, analysis in sorted(all_analyses.items()):
            model_name = analysis['model_name']
            f.write(f"| {model_name} | {key.split('_')[-1]} | "
                   f"{analysis['collapse_rate']:.1f}% | "
                   f"{analysis['answer_rate']:.1f}% | "
                   f"{analysis['avg_words']:.0f} | "
                   f"{analysis['avg_wait']:.1f} | "
                   f"{analysis['avg_diversity']:.3f} |\n")

        f.write("\n")

        # 상세 분석 섹션
        f.write("## 2. 모델별 상세 분석\n\n")

        for key, analysis in sorted(all_analyses.items()):
            model_name = analysis['model_name']
            dataset = key.split('_')[-1]

            f.write(f"### 2.{list(all_analyses.keys()).index(key)+1} {model_name} ({dataset})\n\n")

            f.write("#### 붕괴 유형 분포\n\n")
            f.write("| 유형 | 개수 | 비율 |\n")
            f.write("|------|------|------|\n")

            total = analysis['total_problems']
            for collapse_type, count in sorted(analysis['collapse_counts'].items(),
                                               key=lambda x: -x[1]):
                rate = count / total * 100 if total > 0 else 0
                f.write(f"| {collapse_type} | {count} | {rate:.1f}% |\n")

            f.write("\n")

            # 문제 유형별 붕괴율
            f.write("#### 문제 유형별 붕괴율\n\n")
            f.write("| 문제 유형 | 전체 | 붕괴 | 붕괴율 |\n")
            f.write("|----------|------|------|-------|\n")

            for ptype, counts in sorted(analysis['problems_by_type'].items(),
                                        key=lambda x: -x[1]['collapsed']):
                if counts['total'] > 0:
                    rate = counts['collapsed'] / counts['total'] * 100
                    f.write(f"| {ptype} | {counts['total']} | {counts['collapsed']} | {rate:.1f}% |\n")

            f.write("\n")

            # 붕괴 예시
            if analysis['collapse_examples']:
                f.write("#### 붕괴 예시\n\n")
                for ex in analysis['collapse_examples'][:3]:
                    f.write(f"**Index {ex['index']}** ({ex['collapse_type']})\n")
                    f.write(f"- 문제: {ex['problem']}...\n")
                    f.write(f"- 단어 수: {ex['words']}, 다양성: {ex['diversity']}, Wait: {ex['wait_count']}\n")
                    f.write(f"- 답변: {ex['answer']}\n")
                    f.write(f"- 응답 끝:\n```\n{ex['text_end'][:300]}...\n```\n\n")

            f.write("\n---\n\n")

        # 결론
        f.write("## 3. 결론 및 권장사항\n\n")
        f.write("### 3.1 양자화 방법별 안정성 비교\n\n")
        f.write("분석 결과에 따른 안정성 순위:\n\n")

        # 붕괴율 기준 정렬
        math500_analyses = {k: v for k, v in all_analyses.items() if 'MATH-500' in k}
        sorted_by_collapse = sorted(math500_analyses.items(),
                                    key=lambda x: x[1]['collapse_rate'])

        f.write("| 순위 | 모델 | 붕괴율 | 평가 |\n")
        f.write("|------|------|--------|------|\n")
        for i, (key, analysis) in enumerate(sorted_by_collapse, 1):
            rate = analysis['collapse_rate']
            if rate < 5:
                eval_text = "✅ 매우 안정"
            elif rate < 15:
                eval_text = "✅ 안정"
            elif rate < 30:
                eval_text = "⚠️ 주의 필요"
            else:
                eval_text = "❌ 불안정"
            f.write(f"| {i} | {analysis['model_name']} | {rate:.1f}% | {eval_text} |\n")

        f.write("\n### 3.2 실용적 권장사항\n\n")
        f.write("1. **SmoothQuant (W8A8KV8)**: Weight/Activation/KV Cache 모두 8-bit 양자화\n")
        f.write("2. **KVQuant* (KV4)**: KV Cache만 4-bit 양자화\n")
        f.write("3. **KVQuant* (KV3)**: KV Cache만 3-bit 양자화\n\n")

        f.write("---\n\n")
        f.write("**보고서 생성**: scripts/analysis/smoothquant_kvquant_collapse_analysis.py\n")

    print(f"보고서 생성 완료: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='SmoothQuant/KVQuant 붕괴 분석')
    parser.add_argument('--inference_dir', type=str,
                        default='./outputs/inference',
                        help='추론 결과 디렉토리')
    parser.add_argument('--output_dir', type=str,
                        default='./reports',
                        help='보고서 출력 디렉토리')
    args = parser.parse_args()

    # 분석 대상 모델 설정
    models = {
        'baseline': 'DeepSeek-R1-Distill-Qwen-1.5B-seed42',
        'smoothquant': 'DeepSeek-R1-Distill-Qwen-1.5B-smoothquant-w8a8kv8-tp1-seed42',
        'kvquant_kv4': 'DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1-seed42',
        'kvquant_kv3': 'DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv3-tp1-seed42',
        'awq_3bit': 'DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1-seed42',
        'gptq_3bit': 'DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1-seed42',
        'awq_4bit': 'DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1-seed42',
        'gptq_4bit': 'DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1-seed42',
    }

    datasets = ['MATH-500', 'AIME-90']

    all_analyses = {}

    print("=" * 80)
    print("SmoothQuant/KVQuant 붕괴 분석 시작")
    print("=" * 80)

    for model_key, model_dir in models.items():
        for dataset in datasets:
            filepath = os.path.join(args.inference_dir, model_dir, f"{dataset}.jsonl")

            if not os.path.exists(filepath):
                print(f"[SKIP] 파일 없음: {filepath}")
                continue

            print(f"\n[분석 중] {model_key} - {dataset}")

            try:
                results = load_inference_results(filepath)
                analysis = analyze_model_results(results, model_key)

                key = f"{model_key}_{dataset}"
                all_analyses[key] = analysis

                print(f"  - 총 문제: {analysis['total_problems']}")
                print(f"  - 붕괴율: {analysis['collapse_rate']:.1f}%")
                print(f"  - 답변 생성률: {analysis['answer_rate']:.1f}%")
                print(f"  - 평균 다양성: {analysis['avg_diversity']:.3f}")

            except Exception as e:
                print(f"  [ERROR] {e}")

    # 보고서 생성
    print("\n" + "=" * 80)
    print("보고서 생성 중...")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = generate_report(all_analyses, args.output_dir)

    # JSON 결과 저장
    json_path = os.path.join(args.output_dir, 'smoothquant_kvquant_analysis.json')

    # defaultdict를 일반 dict로 변환
    serializable_analyses = {}
    for key, analysis in all_analyses.items():
        serializable_analyses[key] = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in analysis.items()
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_analyses, f, ensure_ascii=False, indent=2)

    print(f"JSON 결과 저장: {json_path}")

    # 콘솔에 요약 출력
    print("\n" + "=" * 80)
    print("분석 결과 요약")
    print("=" * 80)

    print("\n[MATH-500 붕괴율 비교]")
    print(f"{'모델':<20} {'붕괴율':<10} {'답변생성률':<12} {'평균Wait':<10} {'평균다양성':<10}")
    print("-" * 62)

    for key, analysis in sorted(all_analyses.items()):
        if 'MATH-500' in key:
            model_name = analysis['model_name']
            print(f"{model_name:<20} {analysis['collapse_rate']:<10.1f}% "
                  f"{analysis['answer_rate']:<12.1f}% "
                  f"{analysis['avg_wait']:<10.1f} "
                  f"{analysis['avg_diversity']:<10.3f}")


if __name__ == '__main__':
    main()

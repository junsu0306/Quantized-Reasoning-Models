"""
3-bit vs 4-bit 양자화 상세 비교 분석

AWQ와 GPTQ 각각에 대해 3-bit과 4-bit의 차이를 심층 분석:
1. 4-bit에서 정답, 3-bit에서 오답인 케이스 분석
2. 문제 유형, 복잡도, 특성 분석
3. 응답 패턴 비교 (길이, 반복, 토큰 다양성 등)
4. 실패 원인 분류
5. 구체적 예시 추출
"""

import os
import json
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path
import pandas as pd


class Bit3vs4Comparator:
    """3-bit vs 4-bit 비교 분석 클래스"""

    def __init__(self, inference_dir: str, seed: int, output_dir: str):
        self.inference_dir = inference_dir
        self.seed = seed
        self.output_dir = output_dir

        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_dir, "intermediate")).mkdir(exist_ok=True)

    def load_responses(self, model_name: str, dataset: str) -> List[Dict]:
        """모델의 응답 로드"""
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
        """답변 추출"""
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        gsm_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
        if gsm_match:
            return gsm_match.group(1).strip()

        return None

    def detect_repetition_markers(self, text: str) -> Dict[str, int]:
        """반복 마커 카운트"""
        return {
            'wait': len(re.findall(r'\bWait,?\b', text, re.IGNORECASE)),
            'hmm': len(re.findall(r'\bHmm,?\b', text, re.IGNORECASE)),
            'perhaps': len(re.findall(r'\bPerhaps\b', text, re.IGNORECASE)),
            'let_me': len(re.findall(r'\bLet me\b', text, re.IGNORECASE)),
            'so': len(re.findall(r'\bSo,\b', text)),
            'but': len(re.findall(r'\bBut,?\b', text, re.IGNORECASE)),
        }

    def calculate_token_diversity(self, text: str) -> float:
        """토큰 다양성 계산"""
        tokens = text.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def detect_loop_pattern(self, text: str) -> Dict[str, Any]:
        """루프 패턴 감지"""
        # 숫자 반복
        has_number_loop = bool(re.search(r'(\d)\s*[.,]?\s*(\1\s*[.,]?\s*){10,}', text))

        # 구문 반복
        has_phrase_repeat = bool(re.search(r'([^.]{30,80}\.)\s*(\1\s*){3,}', text))

        # 단어 연속 반복
        has_word_repeat = bool(re.search(r'\b(\w{4,})\s+\1\s+\1\s+\1', text))

        markers = self.detect_repetition_markers(text)

        # 루프 유형 분류
        loop_type = "no_loop"
        if markers['wait'] > 1000:
            loop_type = "severe_wait_loop"
        elif markers['wait'] > 100:
            loop_type = "moderate_wait_loop"
        elif markers['perhaps'] > 100:
            loop_type = "perhaps_loop"
        elif markers['let_me'] > 50:
            loop_type = "let_me_loop"
        elif has_number_loop:
            loop_type = "number_loop"
        elif has_phrase_repeat:
            loop_type = "phrase_repeat"
        elif has_word_repeat:
            loop_type = "word_repeat"
        elif sum(markers.values()) > 100:
            loop_type = "mixed_loop"

        return {
            'loop_type': loop_type,
            'has_number_loop': has_number_loop,
            'has_phrase_repeat': has_phrase_repeat,
            'has_word_repeat': has_word_repeat,
            'total_markers': sum(markers.values())
        }

    def analyze_problem_complexity(self, text: str, problem: str) -> Dict[str, Any]:
        """문제 복잡도 분석"""
        # 문제 텍스트 길이
        problem_length = len(problem.split()) if problem else 0

        # 수식 개수
        math_count = len(re.findall(r'\$[^$]+\$', problem)) if problem else 0

        # 특수 키워드 (boolean flags - can overlap)
        # Use \b for word boundaries to avoid false matches (e.g., "since" matching "sin")
        has_geometry = bool(re.search(r'\b(triangle|circle|angle|rectangle|square|polygon)\b', problem, re.IGNORECASE)) if problem else False
        has_algebra = bool(re.search(r'\b(equation|polynomial|coefficient)\b', problem, re.IGNORECASE)) if problem else False
        has_calculus = bool(re.search(r'\b(derivative|integral|limit)\b', problem, re.IGNORECASE)) if problem else False
        has_trig = bool(re.search(r'\b(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sine|cosine|tangent)\b', problem, re.IGNORECASE)) if problem else False
        has_number_theory = bool(re.search(r'\b(prime|divisor|gcd|lcm|modulo|remainder|divisible)\b', problem, re.IGNORECASE)) if problem else False
        has_combinatorics = bool(re.search(r'\b(combination|permutation|arrange|choose)\b', problem, re.IGNORECASE)) if problem else False
        has_probability = bool(re.search(r'\b(probability|chance|random|expected)\b', problem, re.IGNORECASE)) if problem else False

        # Primary problem type (mutually exclusive - first match wins)
        primary_type = 'other'
        if has_trig:
            primary_type = 'trigonometry'
        elif has_geometry:
            primary_type = 'geometry'
        elif has_calculus:
            primary_type = 'calculus'
        elif has_number_theory:
            primary_type = 'number_theory'
        elif has_combinatorics:
            primary_type = 'combinatorics'
        elif has_probability:
            primary_type = 'probability'
        elif has_algebra:
            primary_type = 'algebra'

        return {
            'problem_word_count': problem_length,
            'math_expression_count': math_count,
            'has_geometry': has_geometry,
            'has_algebra': has_algebra,
            'has_calculus': has_calculus,
            'has_trigonometry': has_trig,
            'has_number_theory': has_number_theory,
            'has_combinatorics': has_combinatorics,
            'has_probability': has_probability,
            'primary_problem_type': primary_type
        }

    def compare_models(self, model_3bit: str, model_4bit: str, dataset: str) -> Dict[str, Any]:
        """3-bit vs 4-bit 비교"""
        print(f"\n{'='*80}")
        print(f"비교: {model_3bit} vs {model_4bit}")
        print(f"데이터셋: {dataset}")
        print(f"{'='*80}")

        # 데이터 로드
        data_3bit = self.load_responses(model_3bit, dataset)
        data_4bit = self.load_responses(model_4bit, dataset)

        if not data_3bit or not data_4bit:
            print("  ⚠ 데이터 로드 실패")
            return None

        # 각 문제별 비교
        comparison_results = []

        for idx in range(len(data_3bit)):
            item_3bit = data_3bit[idx]
            item_4bit = data_4bit[idx]

            # 정답 여부
            metrics_3bit = item_3bit.get('metrics', {})
            metrics_4bit = item_4bit.get('metrics', {})

            correct_3bit = metrics_3bit.get('is_correct', metrics_3bit.get('extractive_match', 0.0)) == 1.0
            correct_4bit = metrics_4bit.get('is_correct', metrics_4bit.get('extractive_match', 0.0)) == 1.0

            text_3bit = item_3bit['generated_text']
            text_4bit = item_4bit['generated_text']

            problem = item_3bit.get('full_prompt', '')
            gold = item_3bit.get('gold', 'N/A')

            # 응답 분석
            markers_3bit = self.detect_repetition_markers(text_3bit)
            markers_4bit = self.detect_repetition_markers(text_4bit)

            loop_3bit = self.detect_loop_pattern(text_3bit)
            loop_4bit = self.detect_loop_pattern(text_4bit)

            complexity = self.analyze_problem_complexity(text_4bit, problem)

            result = {
                'idx': idx,
                'gold_answer': gold,
                'correct_3bit': correct_3bit,
                'correct_4bit': correct_4bit,

                # 비교 카테고리
                'both_correct': correct_3bit and correct_4bit,
                'both_wrong': not correct_3bit and not correct_4bit,
                'only_4bit_correct': correct_4bit and not correct_3bit,  # 핵심!
                'only_3bit_correct': correct_3bit and not correct_4bit,

                # 3-bit 응답 특성
                'word_count_3bit': len(text_3bit.split()),
                'char_count_3bit': len(text_3bit),
                'wait_count_3bit': markers_3bit['wait'],
                'perhaps_count_3bit': markers_3bit['perhaps'],
                'total_markers_3bit': sum(markers_3bit.values()),
                'token_diversity_3bit': self.calculate_token_diversity(text_3bit),
                'has_boxed_3bit': bool(re.search(r'\\boxed\{', text_3bit)),
                'loop_type_3bit': loop_3bit['loop_type'],

                # 4-bit 응답 특성
                'word_count_4bit': len(text_4bit.split()),
                'char_count_4bit': len(text_4bit),
                'wait_count_4bit': markers_4bit['wait'],
                'perhaps_count_4bit': markers_4bit['perhaps'],
                'total_markers_4bit': sum(markers_4bit.values()),
                'token_diversity_4bit': self.calculate_token_diversity(text_4bit),
                'has_boxed_4bit': bool(re.search(r'\\boxed\{', text_4bit)),
                'loop_type_4bit': loop_4bit['loop_type'],

                # 문제 복잡도
                **complexity,

                # 원본 텍스트 (샘플링)
                'text_3bit_preview': text_3bit[:500],
                'text_4bit_preview': text_4bit[:500],
                'text_3bit_last_500': text_3bit[-500:],
                'text_4bit_last_500': text_4bit[-500:],

                # 전체 텍스트 (상세 분석용)
                'full_text_3bit': text_3bit,
                'full_text_4bit': text_4bit,
                'problem': problem
            }

            comparison_results.append(result)

        # 통계 집계
        df = pd.DataFrame(comparison_results)

        stats = {
            'model_3bit': model_3bit,
            'model_4bit': model_4bit,
            'dataset': dataset,
            'total_samples': len(df),

            # 정확도
            'accuracy_3bit': df['correct_3bit'].sum() / len(df),
            'accuracy_4bit': df['correct_4bit'].sum() / len(df),

            # 비교 카테고리
            'both_correct': df['both_correct'].sum(),
            'both_wrong': df['both_wrong'].sum(),
            'only_4bit_correct': df['only_4bit_correct'].sum(),
            'only_3bit_correct': df['only_3bit_correct'].sum(),

            # 상세 결과
            'detailed_results': comparison_results
        }

        return stats

    def analyze_degradation_cases(self, stats: Dict) -> Dict[str, Any]:
        """3-bit에서만 실패한 케이스 심층 분석"""
        df = pd.DataFrame(stats['detailed_results'])

        # 4-bit 정답, 3-bit 오답
        degraded = df[df['only_4bit_correct'] == True].copy()

        if len(degraded) == 0:
            return {'error': '해당 케이스 없음'}

        print(f"\n{'='*80}")
        print(f"3-bit 성능 저하 케이스: {len(degraded)}개")
        print(f"{'='*80}\n")

        # 1. 문제 유형별 분포 (overlapping - 중복 포함)
        problem_types_overlapping = {
            'geometry': int(degraded['has_geometry'].sum()),
            'algebra': int(degraded['has_algebra'].sum()),
            'calculus': int(degraded['has_calculus'].sum()),
            'trigonometry': int(degraded['has_trigonometry'].sum()),
            'number_theory': int(degraded['has_number_theory'].sum()),
            'combinatorics': int(degraded['has_combinatorics'].sum()),
            'probability': int(degraded['has_probability'].sum())
        }

        # 1-2. Primary problem type (mutually exclusive - 상호 배타적)
        primary_type_dist = degraded['primary_problem_type'].value_counts().to_dict()

        # 2. 루프 패턴 분포
        loop_distribution = degraded['loop_type_3bit'].value_counts().to_dict()

        # 3. 응답 길이 분석
        length_stats = {
            'avg_word_count_3bit': degraded['word_count_3bit'].mean(),
            'avg_word_count_4bit': degraded['word_count_4bit'].mean(),
            'median_word_count_3bit': degraded['word_count_3bit'].median(),
            'median_word_count_4bit': degraded['word_count_4bit'].median(),
            'length_ratio': degraded['word_count_3bit'].mean() / degraded['word_count_4bit'].mean() if degraded['word_count_4bit'].mean() > 0 else 0
        }

        # 4. 반복 마커 분석
        marker_stats = {
            'avg_wait_3bit': degraded['wait_count_3bit'].mean(),
            'avg_wait_4bit': degraded['wait_count_4bit'].mean(),
            'avg_perhaps_3bit': degraded['perhaps_count_3bit'].mean(),
            'avg_perhaps_4bit': degraded['perhaps_count_4bit'].mean(),
            'avg_total_markers_3bit': degraded['total_markers_3bit'].mean(),
            'avg_total_markers_4bit': degraded['total_markers_4bit'].mean()
        }

        # 5. 토큰 다양성
        diversity_stats = {
            'avg_diversity_3bit': degraded['token_diversity_3bit'].mean(),
            'avg_diversity_4bit': degraded['token_diversity_4bit'].mean(),
            'diversity_drop': (degraded['token_diversity_4bit'].mean() - degraded['token_diversity_3bit'].mean())
        }

        # 6. Answer 생성 실패
        answer_stats = {
            'has_boxed_3bit_rate': degraded['has_boxed_3bit'].sum() / len(degraded),
            'has_boxed_4bit_rate': degraded['has_boxed_4bit'].sum() / len(degraded)
        }

        # 7. 문제 복잡도별 분포
        complexity_bins = pd.cut(degraded['word_count_4bit'],
                                 bins=[0, 1000, 2000, 3000, 5000, 10000, 30000],
                                 labels=['<1k', '1-2k', '2-3k', '3-5k', '5-10k', '>10k'])
        complexity_distribution = complexity_bins.value_counts().to_dict()

        # 8. 상위 10개 worst case
        degraded_sorted = degraded.sort_values('word_count_3bit', ascending=False)
        worst_cases = []
        for idx, row in degraded_sorted.head(10).iterrows():
            worst_cases.append({
                'problem_idx': int(row['idx']),
                'word_count_3bit': int(row['word_count_3bit']),
                'word_count_4bit': int(row['word_count_4bit']),
                'wait_count_3bit': int(row['wait_count_3bit']),
                'loop_type_3bit': row['loop_type_3bit'],
                'primary_problem_type': row['primary_problem_type'],
                'has_geometry': bool(row['has_geometry']),
                'has_trigonometry': bool(row['has_trigonometry'])
            })

        return {
            'count': len(degraded),
            'problem_types_overlapping': problem_types_overlapping,
            'primary_problem_types': primary_type_dist,
            'loop_distribution': loop_distribution,
            'length_stats': length_stats,
            'marker_stats': marker_stats,
            'diversity_stats': diversity_stats,
            'answer_stats': answer_stats,
            'complexity_distribution': complexity_distribution,
            'worst_cases': worst_cases,
            'degraded_dataframe': degraded
        }

    def generate_detailed_examples(self, degradation_analysis: Dict, stats: Dict, n_examples: int = 5) -> List[Dict]:
        """상세 예시 생성"""
        if 'degraded_dataframe' not in degradation_analysis:
            return []

        df = degradation_analysis['degraded_dataframe']

        examples = []

        # 다양한 케이스 선택
        # 1. 심각한 루프 케이스
        severe_loop = df[df['loop_type_3bit'].str.contains('severe', na=False)]
        if len(severe_loop) > 0:
            case = severe_loop.iloc[0]
            examples.append({
                'type': 'Severe Loop',
                'problem_idx': case['idx'],
                'gold_answer': case['gold_answer'],
                'word_count_3bit': case['word_count_3bit'],
                'word_count_4bit': case['word_count_4bit'],
                'wait_count_3bit': case['wait_count_3bit'],
                'loop_type': case['loop_type_3bit'],
                'text_3bit_start': case['text_3bit_preview'],
                'text_3bit_end': case['text_3bit_last_500'],
                'text_4bit_start': case['text_4bit_preview'],
                'text_4bit_end': case['text_4bit_last_500'],
                'full_problem': case['problem']
            })

        # 2. 기하학 문제
        geometry = df[df['has_geometry'] == True]
        if len(geometry) > 0:
            case = geometry.iloc[0]
            examples.append({
                'type': 'Geometry Problem',
                'problem_idx': case['idx'],
                'gold_answer': case['gold_answer'],
                'word_count_3bit': case['word_count_3bit'],
                'word_count_4bit': case['word_count_4bit'],
                'wait_count_3bit': case['wait_count_3bit'],
                'loop_type': case['loop_type_3bit'],
                'text_3bit_start': case['text_3bit_preview'],
                'text_3bit_end': case['text_3bit_last_500'],
                'text_4bit_start': case['text_4bit_preview'],
                'text_4bit_end': case['text_4bit_last_500'],
                'full_problem': case['problem']
            })

        # 3. 삼각함수 문제
        trig = df[df['has_trigonometry'] == True]
        if len(trig) > 0:
            case = trig.iloc[0]
            examples.append({
                'type': 'Trigonometry Problem',
                'problem_idx': case['idx'],
                'gold_answer': case['gold_answer'],
                'word_count_3bit': case['word_count_3bit'],
                'word_count_4bit': case['word_count_4bit'],
                'wait_count_3bit': case['wait_count_3bit'],
                'loop_type': case['loop_type_3bit'],
                'text_3bit_start': case['text_3bit_preview'],
                'text_3bit_end': case['text_3bit_last_500'],
                'text_4bit_start': case['text_4bit_preview'],
                'text_4bit_end': case['text_4bit_last_500'],
                'full_problem': case['problem']
            })

        # 4. Perhaps Loop
        perhaps = df[df['loop_type_3bit'] == 'perhaps_loop']
        if len(perhaps) > 0:
            case = perhaps.iloc[0]
            examples.append({
                'type': 'Perhaps Loop',
                'problem_idx': case['idx'],
                'gold_answer': case['gold_answer'],
                'word_count_3bit': case['word_count_3bit'],
                'word_count_4bit': case['word_count_4bit'],
                'perhaps_count_3bit': case['perhaps_count_3bit'],
                'loop_type': case['loop_type_3bit'],
                'text_3bit_start': case['text_3bit_preview'],
                'text_3bit_end': case['text_3bit_last_500'],
                'text_4bit_start': case['text_4bit_preview'],
                'text_4bit_end': case['text_4bit_last_500'],
                'full_problem': case['problem']
            })

        # 5. Number Loop
        number = df[df['loop_type_3bit'] == 'number_loop']
        if len(number) > 0:
            case = number.iloc[0]
            examples.append({
                'type': 'Number Loop',
                'problem_idx': case['idx'],
                'gold_answer': case['gold_answer'],
                'word_count_3bit': case['word_count_3bit'],
                'word_count_4bit': case['word_count_4bit'],
                'wait_count_3bit': case['wait_count_3bit'],
                'loop_type': case['loop_type_3bit'],
                'text_3bit_start': case['text_3bit_preview'],
                'text_3bit_end': case['text_3bit_last_500'],
                'text_4bit_start': case['text_4bit_preview'],
                'text_4bit_end': case['text_4bit_last_500'],
                'full_problem': case['problem']
            })

        return examples[:n_examples]

    def convert_to_python_types(self, obj):
        """numpy 타입을 Python 타입으로 변환"""
        if isinstance(obj, dict):
            return {k: self.convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results(self, quant_method: str, stats: Dict, degradation: Dict, examples: List[Dict]):
        """결과 저장"""
        base_name = f"{quant_method}_3bit_vs_4bit_{stats['dataset']}"

        # 1. 전체 통계 (JSON)
        stats_copy = stats.copy()
        if 'detailed_results' in stats_copy:
            del stats_copy['detailed_results']  # 너무 크므로 제외

        stats_copy = self.convert_to_python_types(stats_copy)

        with open(os.path.join(self.output_dir, "intermediate", f"{base_name}_stats.json"), 'w') as f:
            json.dump(stats_copy, f, indent=2, ensure_ascii=False)

        # 2. Degradation 분석 (JSON)
        degradation_copy = degradation.copy()
        if 'degraded_dataframe' in degradation_copy:
            del degradation_copy['degraded_dataframe']

        degradation_copy = self.convert_to_python_types(degradation_copy)

        with open(os.path.join(self.output_dir, "intermediate", f"{base_name}_degradation.json"), 'w') as f:
            json.dump(degradation_copy, f, indent=2, ensure_ascii=False)

        # 3. 상세 예시 (JSON)
        examples = self.convert_to_python_types(examples)

        with open(os.path.join(self.output_dir, "intermediate", f"{base_name}_examples.json"), 'w') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        # 4. DataFrame (CSV)
        df = pd.DataFrame(stats['detailed_results'])
        df.to_csv(os.path.join(self.output_dir, "intermediate", f"{base_name}_full.csv"), index=False)

        # 5. Degraded cases only (CSV)
        if 'degraded_dataframe' in degradation:
            degraded_df = degradation['degraded_dataframe']
            degraded_df.to_csv(os.path.join(self.output_dir, "intermediate", f"{base_name}_degraded_only.csv"), index=False)

        print(f"\n✓ 결과 저장 완료: {self.output_dir}/intermediate/{base_name}_*")


def main():
    parser = argparse.ArgumentParser(description='3-bit vs 4-bit 상세 비교 분석')
    parser.add_argument('--inference_dir', type=str, default='./outputs/inference')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./reports/3bit_vs_4bit_analysis')
    parser.add_argument('--datasets', type=str, nargs='+', default=['MATH-500', 'AIME-90'])

    args = parser.parse_args()

    comparator = Bit3vs4Comparator(args.inference_dir, args.seed, args.output_dir)

    all_results = {}

    # AWQ 비교
    print("\n" + "="*80)
    print("AWQ 3-bit vs 4-bit 분석")
    print("="*80)

    for dataset in args.datasets:
        stats_awq = comparator.compare_models(
            'DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1',
            'DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1',
            dataset
        )

        if stats_awq:
            degradation_awq = comparator.analyze_degradation_cases(stats_awq)
            examples_awq = comparator.generate_detailed_examples(degradation_awq, stats_awq, n_examples=5)
            comparator.save_results('AWQ', stats_awq, degradation_awq, examples_awq)

            all_results[f'AWQ_{dataset}'] = {
                'stats': stats_awq,
                'degradation': degradation_awq,
                'examples': examples_awq
            }

    # GPTQ 비교
    print("\n" + "="*80)
    print("GPTQ 3-bit vs 4-bit 분석")
    print("="*80)

    for dataset in args.datasets:
        stats_gptq = comparator.compare_models(
            'DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1',
            'DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1',
            dataset
        )

        if stats_gptq:
            degradation_gptq = comparator.analyze_degradation_cases(stats_gptq)
            examples_gptq = comparator.generate_detailed_examples(degradation_gptq, stats_gptq, n_examples=5)
            comparator.save_results('GPTQ', stats_gptq, degradation_gptq, examples_gptq)

            all_results[f'GPTQ_{dataset}'] = {
                'stats': stats_gptq,
                'degradation': degradation_gptq,
                'examples': examples_gptq
            }

    print("\n" + "="*80)
    print("전체 분석 완료!")
    print("="*80)
    print(f"\n결과 위치: {args.output_dir}/")


if __name__ == "__main__":
    main()

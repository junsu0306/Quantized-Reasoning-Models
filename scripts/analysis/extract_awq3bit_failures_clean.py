#!/usr/bin/env python3
"""
AWQ 3-bit만 실패한 164개 케이스 - 응답 중심 정리본
- 통계 제거, 문제 요약, 응답 상세히 표시
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List


def summarize_problem(problem_text: str) -> str:
    """문제를 한 줄로 요약"""
    # 태그 제거
    problem = re.sub(r'<[^>]+>', '', problem_text)
    problem = problem.strip()

    # 첫 200자만
    if len(problem) > 200:
        problem = problem[:200] + "..."

    # 문제 유형 키워드 감지
    problem_type = []
    if re.search(r'\b(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan)\b', problem, re.IGNORECASE):
        problem_type.append("삼각함수")
    if re.search(r'\b(triangle|circle|angle|rectangle|square|polygon)\b', problem, re.IGNORECASE):
        problem_type.append("기하")
    if re.search(r'\b(prime|divisor|gcd|lcm|modulo)\b', problem, re.IGNORECASE):
        problem_type.append("정수론")
    if re.search(r'\b(probability|chance|random|expected)\b', problem, re.IGNORECASE):
        problem_type.append("확률")
    if re.search(r'\b(combination|permutation|choose)\b', problem, re.IGNORECASE):
        problem_type.append("조합론")
    if re.search(r'\b(polynomial|equation|coefficient)\b', problem, re.IGNORECASE):
        problem_type.append("대수")

    type_str = f"[{', '.join(problem_type)}] " if problem_type else "[일반 수학] "

    return type_str + problem


def clean_response_text(text: str) -> str:
    """응답 텍스트 정리 (태그 제거)"""
    # 시작 태그들 제거
    text = re.sub(r'<｜begin▁of▁sentence｜>', '', text)
    text = re.sub(r'<｜User｜>', '', text)
    text = re.sub(r'<｜Assistant｜>', '', text)
    text = re.sub(r'<think>', '\n[추론 시작]', text)
    text = re.sub(r'</think>', '[추론 끝]\n', text)
    text = re.sub(r'<｜end▁of▁sentence｜>', '', text)

    return text.strip()


def detect_repetition_pattern(text: str) -> Dict[str, Any]:
    """반복 패턴 감지"""
    word_count = len(text.split())

    # Wait 패턴
    wait_count = len(re.findall(r'\bWait\b', text, re.IGNORECASE))

    # Perhaps 패턴
    perhaps_count = len(re.findall(r'\bPerhaps\b', text, re.IGNORECASE))

    # Let me 패턴
    letme_count = len(re.findall(r'\bLet me\b', text, re.IGNORECASE))

    # Number count
    numbers = re.findall(r'\b\d+\b', text)

    # 반복 판단
    is_repetition = False
    pattern_desc = None

    if wait_count > 100:
        is_repetition = True
        pattern_desc = f"⚠️ SEVERE REPETITION: 'Wait' 패턴 {wait_count}회 반복 (총 {word_count:,} 단어)"
    elif wait_count > 30:
        is_repetition = True
        pattern_desc = f"⚠️ MODERATE REPETITION: 'Wait' 패턴 {wait_count}회 반복 (총 {word_count:,} 단어)"
    elif perhaps_count > 50:
        is_repetition = True
        pattern_desc = f"⚠️ REPETITION: 'Perhaps' 패턴 {perhaps_count}회 반복 (총 {word_count:,} 단어)"
    elif letme_count > 100:
        is_repetition = True
        pattern_desc = f"⚠️ REPETITION: 'Let me' 패턴 {letme_count}회 반복 (총 {word_count:,} 단어)"
    elif len(numbers) > 500:
        is_repetition = True
        pattern_desc = f"⚠️ NUMBER LOOP: 숫자 반복 (총 {word_count:,} 단어, {len(numbers)}개 숫자)"
    elif word_count > 10000:
        pattern_desc = f"⚠️ VERY LONG: 총 {word_count:,} 단어"

    return {
        'is_repetition': is_repetition,
        'pattern_desc': pattern_desc,
        'word_count': word_count
    }


def format_response(text: str, max_length: int = 15000) -> str:
    """응답 포맷팅 (반복이면 축약, 아니면 전체 또는 제한)"""
    pattern = detect_repetition_pattern(text)
    cleaned = clean_response_text(text)

    if pattern['is_repetition']:
        # 반복이면 앞 2000자 + 설명 + 뒤 1000자
        preview = cleaned[:2000]
        tail = cleaned[-1000:]

        return (
            f"{pattern['pattern_desc']}\n\n"
            f"{'='*80}\n"
            f"[처음 2000자]\n\n{preview}\n\n"
            f"{'='*80}\n"
            f"[... 중간 생략 ...]\n"
            f"{'='*80}\n\n"
            f"[마지막 1000자]\n\n{tail}"
        )
    else:
        # 반복 아니면 그대로 (단, max_length 제한)
        if len(cleaned) > max_length:
            if pattern['pattern_desc']:
                return f"{pattern['pattern_desc']}\n\n{'='*80}\n\n{cleaned[:max_length]}\n\n{'='*80}\n[... {len(cleaned) - max_length} 자 생략 ...]"
            return f"{cleaned[:max_length]}\n\n{'='*80}\n[... {len(cleaned) - max_length} 자 생략 ...]"
        else:
            if pattern['pattern_desc']:
                return f"{pattern['pattern_desc']}\n\n{'='*80}\n\n{cleaned}"
            return cleaned


def load_model_data(model_name: str, dataset: str) -> List[Dict]:
    """특정 모델의 데이터 로드"""
    path = f"outputs/inference/{model_name}-seed42/{dataset}.jsonl"
    with open(path) as f:
        return json.load(f)


def extract_clean_comparison():
    """깔끔한 비교 JSON 생성"""

    print("📂 데이터 로딩...")

    # 4개 모델 데이터 로드
    awq_3bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1", "MATH-500")
    awq_4bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1", "MATH-500")
    gptq_3bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1", "MATH-500")
    gptq_4bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1", "MATH-500")

    print(f"✓ 로드 완료: {len(awq_3bit)} 문제\n")

    results = []

    for idx in range(len(awq_3bit)):
        # 정답 여부 확인
        metrics_awq3 = awq_3bit[idx].get('metrics', {})
        metrics_awq4 = awq_4bit[idx].get('metrics', {})
        metrics_gptq3 = gptq_3bit[idx].get('metrics', {})
        metrics_gptq4 = gptq_4bit[idx].get('metrics', {})

        correct_awq3 = metrics_awq3.get('is_correct', metrics_awq3.get('extractive_match', 0.0)) == 1.0
        correct_awq4 = metrics_awq4.get('is_correct', metrics_awq4.get('extractive_match', 0.0)) == 1.0
        correct_gptq3 = metrics_gptq3.get('is_correct', metrics_gptq3.get('extractive_match', 0.0)) == 1.0
        correct_gptq4 = metrics_gptq4.get('is_correct', metrics_gptq4.get('extractive_match', 0.0)) == 1.0

        # AWQ 3-bit만 틀린 경우
        if not correct_awq3 and correct_awq4:
            result = {
                "problem_idx": idx,
                "problem_summary": summarize_problem(awq_3bit[idx].get('full_prompt', '')),
                "gold_answer": awq_3bit[idx].get('gold', 'N/A'),

                "correctness": {
                    "awq_3bit": "❌ 오답",
                    "awq_4bit": "✅ 정답",
                    "gptq_3bit": "✅ 정답" if correct_gptq3 else "❌ 오답",
                    "gptq_4bit": "✅ 정답" if correct_gptq4 else "❌ 오답"
                },

                "responses": {
                    "awq_3bit": format_response(awq_3bit[idx]['generated_text']),
                    "awq_4bit": format_response(awq_4bit[idx]['generated_text']),
                    "gptq_3bit": format_response(gptq_3bit[idx]['generated_text']),
                    "gptq_4bit": format_response(gptq_4bit[idx]['generated_text'])
                }
            }

            results.append(result)

            if len(results) % 20 == 0:
                print(f"  처리 중... {len(results)}개 완료")

    print(f"\n✓ 총 {len(results)}개 케이스 추출 완료\n")

    # JSON 저장
    output_dir = Path("reports/3bit_vs_4bit_analysis/")
    output_file = output_dir / "awq_3bit_failures_clean.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 저장 완료: {output_file}")
    print(f"   파일 크기: {output_file.stat().st_size / 1024 / 1024:.1f} MB\n")

    # 통계
    gptq3_also_wrong = sum(1 for r in results if r['correctness']['gptq_3bit'] == "❌ 오답")

    print("📊 요약:")
    print(f"   AWQ 3-bit만 틀린 케이스: {len(results)}개")
    print(f"   이 중 GPTQ 3-bit도 틀림: {gptq3_also_wrong}개 ({gptq3_also_wrong/len(results)*100:.1f}%)")
    print(f"   이 중 GPTQ 3-bit은 정답: {len(results) - gptq3_also_wrong}개 ({(len(results)-gptq3_also_wrong)/len(results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    extract_clean_comparison()

#!/usr/bin/env python3
"""
AWQ 3-bit만 실패한 164개 케이스에 대해 4개 모델(AWQ 3/4bit, GPTQ 3/4bit)의
응답을 비교하기 쉽게 JSON으로 추출

너무 긴 응답(5000+ 단어)은 반복 패턴 요약으로 대체
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List


def detect_repetition_summary(text: str, word_count: int) -> str:
    """반복 패턴 감지 및 요약"""
    if word_count < 5000:
        return text  # 5000 단어 이하는 그대로

    # Wait 패턴
    wait_count = len(re.findall(r'\bWait\b', text, re.IGNORECASE))
    if wait_count > 100:
        preview = text[:500]
        tail = text[-300:]
        return f"[REPETITION DETECTED]\n\n🔁 'Wait' repeated {wait_count} times (total {word_count:,} words)\n\n📝 First 500 chars:\n{preview}\n\n...\n\n📝 Last 300 chars:\n{tail}"

    # Perhaps 패턴
    perhaps_count = len(re.findall(r'\bPerhaps\b', text, re.IGNORECASE))
    if perhaps_count > 50:
        preview = text[:500]
        tail = text[-300:]
        return f"[REPETITION DETECTED]\n\n🔁 'Perhaps' repeated {perhaps_count} times (total {word_count:,} words)\n\n📝 First 500 chars:\n{preview}\n\n...\n\n📝 Last 300 chars:\n{tail}"

    # Let me 패턴
    letme_count = len(re.findall(r'\bLet me\b', text, re.IGNORECASE))
    if letme_count > 100:
        preview = text[:500]
        tail = text[-300:]
        return f"[REPETITION DETECTED]\n\n🔁 'Let me' repeated {letme_count} times (total {word_count:,} words)\n\n📝 First 500 chars:\n{preview}\n\n...\n\n📝 Last 300 chars:\n{tail}"

    # Number loop
    numbers = re.findall(r'\b\d+\b', text)
    if len(numbers) > 500:
        preview = text[:500]
        tail = text[-300:]
        return f"[REPETITION DETECTED]\n\n🔁 Number loop detected (total {word_count:,} words, {len(numbers)} numbers)\n\n📝 First 500 chars:\n{preview}\n\n...\n\n📝 Last 300 chars:\n{tail}"

    # 일반적인 긴 응답
    preview = text[:1000]
    tail = text[-500:]
    return f"[LONG RESPONSE]\n\n📊 Total {word_count:,} words\n\n📝 First 1000 chars:\n{preview}\n\n...\n\n📝 Last 500 chars:\n{tail}"


def load_model_data(model_name: str, dataset: str) -> List[Dict]:
    """특정 모델의 데이터 로드"""
    path = f"outputs/inference/{model_name}-seed42/{dataset}.jsonl"
    with open(path) as f:
        return json.load(f)


def extract_awq3bit_only_failures():
    """AWQ 3-bit만 실패한 케이스 추출"""

    print("📂 데이터 로딩...")

    # 4개 모델 데이터 로드
    awq_3bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1", "MATH-500")
    awq_4bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1", "MATH-500")
    gptq_3bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1", "MATH-500")
    gptq_4bit = load_model_data("DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1", "MATH-500")

    print(f"✓ 로드 완료: {len(awq_3bit)} 문제")

    # AWQ 3-bit만 틀린 케이스 찾기
    comparison_results = []

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

        # AWQ 3-bit만 틀린 경우 (4-bit은 맞음)
        if not correct_awq3 and correct_awq4:
            # 응답 텍스트
            text_awq3 = awq_3bit[idx]['generated_text']
            text_awq4 = awq_4bit[idx]['generated_text']
            text_gptq3 = gptq_3bit[idx]['generated_text']
            text_gptq4 = gptq_4bit[idx]['generated_text']

            # 단어 수
            wc_awq3 = len(text_awq3.split())
            wc_awq4 = len(text_awq4.split())
            wc_gptq3 = len(text_gptq3.split())
            wc_gptq4 = len(text_gptq4.split())

            result = {
                "problem_idx": idx,
                "problem": awq_3bit[idx].get('full_prompt', '')[:1000],  # 문제는 1000자로 제한
                "gold_answer": awq_3bit[idx].get('gold', 'N/A'),

                "awq_3bit": {
                    "answer": detect_repetition_summary(text_awq3, wc_awq3),
                    "correct": correct_awq3,
                    "word_count": wc_awq3,
                    "wait_count": len(re.findall(r'\bWait\b', text_awq3, re.IGNORECASE))
                },

                "awq_4bit": {
                    "answer": detect_repetition_summary(text_awq4, wc_awq4),
                    "correct": correct_awq4,
                    "word_count": wc_awq4
                },

                "gptq_3bit": {
                    "answer": detect_repetition_summary(text_gptq3, wc_gptq3),
                    "correct": correct_gptq3,
                    "word_count": wc_gptq3,
                    "wait_count": len(re.findall(r'\bWait\b', text_gptq3, re.IGNORECASE))
                },

                "gptq_4bit": {
                    "answer": detect_repetition_summary(text_gptq4, wc_gptq4),
                    "correct": correct_gptq4,
                    "word_count": wc_gptq4
                }
            }

            comparison_results.append(result)

    print(f"\n✓ AWQ 3-bit만 틀린 케이스: {len(comparison_results)}개")

    # JSON 저장
    output_dir = Path("reports/3bit_vs_4bit_analysis/")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "awq_3bit_only_failures_comparison.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 저장 완료: {output_file}")
    print(f"   파일 크기: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # 통계
    print("\n📊 통계:")

    # GPTQ 3-bit 정답률
    gptq3_correct = sum(1 for r in comparison_results if r['gptq_3bit']['correct'])
    print(f"   GPTQ 3-bit도 틀림: {len(comparison_results) - gptq3_correct}개 ({(len(comparison_results) - gptq3_correct)/len(comparison_results)*100:.1f}%)")
    print(f"   GPTQ 3-bit은 정답: {gptq3_correct}개 ({gptq3_correct/len(comparison_results)*100:.1f}%)")

    # 반복 문제
    awq3_repetition = sum(1 for r in comparison_results if r['awq_3bit']['wait_count'] > 100)
    gptq3_repetition = sum(1 for r in comparison_results if r['gptq_3bit']['wait_count'] > 100)

    print(f"\n   AWQ 3-bit 심각한 반복 (Wait>100): {awq3_repetition}개 ({awq3_repetition/len(comparison_results)*100:.1f}%)")
    print(f"   GPTQ 3-bit 심각한 반복 (Wait>100): {gptq3_repetition}개 ({gptq3_repetition/len(comparison_results)*100:.1f}%)")

    # 응답 길이
    avg_wc_awq3 = sum(r['awq_3bit']['word_count'] for r in comparison_results) / len(comparison_results)
    avg_wc_gptq3 = sum(r['gptq_3bit']['word_count'] for r in comparison_results) / len(comparison_results)

    print(f"\n   평균 응답 길이:")
    print(f"     AWQ 3-bit:  {avg_wc_awq3:,.0f} 단어")
    print(f"     AWQ 4-bit:  {sum(r['awq_4bit']['word_count'] for r in comparison_results) / len(comparison_results):,.0f} 단어")
    print(f"     GPTQ 3-bit: {avg_wc_gptq3:,.0f} 단어")
    print(f"     GPTQ 4-bit: {sum(r['gptq_4bit']['word_count'] for r in comparison_results) / len(comparison_results):,.0f} 단어")

    return comparison_results


if __name__ == "__main__":
    extract_awq3bit_only_failures()

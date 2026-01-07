#!/usr/bin/env python3
"""실패 원인별 상세 예시 추출"""

import json
import re
import csv
from collections import defaultdict

BASE_DIR = "/Users/junsu/Projects/Quantized-Reasoning-Models"
GPTQ_3BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1-seed42/MATH-500.jsonl"
GPTQ_4BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1-seed42/MATH-500.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/reports/gptq_3bit_error_analysis"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_problem(full_prompt):
    """문제 텍스트만 추출"""
    match = re.search(r'User[｜\|]?>(.+?)(?:Please reason|$)', full_prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return full_prompt[:500]

def count_markers(text):
    return {
        'Wait': len(re.findall(r'\bWait\b', text)),
        'Perhaps': len(re.findall(r'\bPerhaps\b', text)),
        'Let me': len(re.findall(r'Let me', text)),
    }

def extract_boxed_answer(text):
    """\\boxed{} 안의 답 추출"""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1] if matches else None

# 데이터 로드
data_3bit = load_json(GPTQ_3BIT_PATH)
data_4bit = load_json(GPTQ_4BIT_PATH)

# CSV 파일에서 실패 케이스 인덱스 및 원인 로드
failure_cases = {}
with open(f'{OUTPUT_DIR}/all_failures.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row['index'])
        failure_cases[idx] = row['failure_reasons'].split('|')

# 실패 원인별 케이스 분류
reason_examples = defaultdict(list)

for idx, reasons in failure_cases.items():
    d3 = data_3bit[idx]
    d4 = data_4bit[idx]

    text_3bit = d3['generated_text']
    text_4bit = d4['generated_text']

    problem = extract_problem(d3['full_prompt'])
    gold = d3['gold'][0] if isinstance(d3['gold'], list) else d3['gold']

    # 정답 추출
    gold_answer = extract_boxed_answer(gold)
    answer_3bit = extract_boxed_answer(text_3bit)
    answer_4bit = extract_boxed_answer(text_4bit)

    markers_3bit = count_markers(text_3bit)
    markers_4bit = count_markers(text_4bit)

    words_3bit = len(text_3bit.split())
    words_4bit = len(text_4bit.split())

    # 토큰 다양성
    tokens_3bit = text_3bit.split()
    diversity_3bit = len(set(tokens_3bit)) / len(tokens_3bit) if tokens_3bit else 0

    case = {
        'index': idx,
        'problem': problem,
        'gold_answer': gold_answer,
        'answer_3bit': answer_3bit,
        'answer_4bit': answer_4bit,
        'words_3bit': words_3bit,
        'words_4bit': words_4bit,
        'wait_3bit': markers_3bit['Wait'],
        'wait_4bit': markers_4bit['Wait'],
        'diversity_3bit': diversity_3bit,
        'text_3bit_start': text_3bit[:800],
        'text_3bit_end': text_3bit[-800:] if len(text_3bit) > 800 else text_3bit,
        'text_4bit_start': text_4bit[:600],
        'text_4bit_end': text_4bit[-600:] if len(text_4bit) > 600 else text_4bit,
        'reasons': reasons,
    }

    for reason in reasons:
        reason_examples[reason].append(case)

# 각 원인별 대표 케이스 2개씩 선택
selected_examples = {}

# 1. 계산 오류 - 답변은 있지만 값이 틀린 경우
if '계산 오류' in reason_examples:
    cases = [c for c in reason_examples['계산 오류']
             if c['answer_3bit'] and c['answer_3bit'] != c['gold_answer']]
    # 가장 간단한 문제에서 틀린 케이스
    cases.sort(key=lambda x: x['words_4bit'])
    selected_examples['계산 오류'] = cases[:2]

# 2. Wait 무한반복 - Wait 횟수가 많은 케이스
if 'Wait 무한반복' in reason_examples:
    cases = sorted(reason_examples['Wait 무한반복'], key=lambda x: -x['wait_3bit'])
    selected_examples['Wait 무한반복'] = cases[:2]

# 3. 토큰다양성 붕괴 - 다양성이 가장 낮은 케이스
if '토큰다양성 붕괴' in reason_examples:
    cases = sorted(reason_examples['토큰다양성 붕괴'], key=lambda x: x['diversity_3bit'])
    selected_examples['토큰다양성 붕괴'] = cases[:2]

# 4. 응답길이 폭발 - 길이 비율이 가장 큰 케이스
if '응답길이 폭발' in reason_examples:
    cases = sorted(reason_examples['응답길이 폭발'],
                   key=lambda x: -x['words_3bit']/max(x['words_4bit'], 1))
    selected_examples['응답길이 폭발'] = cases[:2]

# 5. 답변 미생성 - boxed 답변이 없는 케이스
if '답변 미생성' in reason_examples:
    cases = [c for c in reason_examples['답변 미생성'] if not c['answer_3bit']]
    cases.sort(key=lambda x: -x['wait_3bit'])
    selected_examples['답변 미생성'] = cases[:2]

# 결과 출력
output = {"examples": selected_examples}
with open(f'{OUTPUT_DIR}/failure_examples_detailed.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("실패 원인별 대표 예시 추출 완료!")
print(f"저장: {OUTPUT_DIR}/failure_examples_detailed.json")

# 요약 출력
for reason, cases in selected_examples.items():
    print(f"\n{'='*60}")
    print(f"[{reason}] - {len(cases)}개 예시")
    for i, c in enumerate(cases):
        print(f"\n  예시 {i+1} (Index: {c['index']})")
        print(f"  문제: {c['problem'][:100]}...")
        print(f"  정답: {c['gold_answer']}")
        print(f"  3-bit 답: {c['answer_3bit']} | 4-bit 답: {c['answer_4bit']}")
        print(f"  3-bit: {c['words_3bit']}단어, Wait {c['wait_3bit']}회, 다양성 {c['diversity_3bit']:.3f}")
        print(f"  4-bit: {c['words_4bit']}단어, Wait {c['wait_4bit']}회")

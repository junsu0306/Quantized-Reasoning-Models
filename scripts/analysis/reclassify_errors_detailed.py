#!/usr/bin/env python3
"""
논문 기반 오류 분류 체계로 GPTQ 3-bit 실패 케이스 재분류

오류 분류 체계 (Quantization Meets Reasoning 논문 기반):
A. 개념 오류 (Conceptual Errors)
   - A1. 개념적 오해 (Conceptual Misunderstanding)
   - A2. 맥락적 간과 (Contextual Oversight)

B. 방법 오류 (Method Errors)
   - B1. 절차적 오류 (Procedural Error)
   - B2. 공식/규칙 오류 (Formula Rule Error)

C. 실행 오류 (Execution Errors)
   - C1. 계산 오류 (Computational Error)
   - C2. 기호 조작 오류 (Symbolic Manipulation Error)

D. 추론 오류 (Reasoning Errors)
   - D1. 논리적 추론 오류 (Logical Reasoning Error)

E. 반복 및 붕괴 (Repetition & Collapse)
   - E1. Wait/Perhaps 무한반복
   - E2. 토큰/숫자/문자 반복 붕괴
   - E3. 문맥 이탈 (완전히 다른 문제 풀이)

우선순위: 개념(A) → 추론(D) → 방법(B) → 실행(C) → 반복/붕괴(E)
"""

import json
import re
import csv
from collections import defaultdict, Counter

BASE_DIR = "/Users/junsu/Projects/Quantized-Reasoning-Models"
GPTQ_3BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1-seed42/MATH-500.jsonl"
GPTQ_4BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1-seed42/MATH-500.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/reports/gptq_3bit_error_analysis"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_problem(full_prompt):
    match = re.search(r'User[｜\|]?>(.+?)(?:Please reason|$)', full_prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return full_prompt[:500]

def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1] if matches else None

def count_repetition_patterns(text):
    """반복 패턴 감지"""
    patterns = {
        'wait_count': len(re.findall(r'\bWait\b', text)),
        'perhaps_count': len(re.findall(r'\bPerhaps\b', text)),
        'let_me_count': len(re.findall(r'Let me', text)),
    }

    # 연속 숫자/문자 반복 감지
    # "2 2 2 2" 또는 "S, S, S, S" 패턴
    number_repeat = re.findall(r'(\d+(?:\s+\d+){10,})', text)
    char_repeat = re.findall(r'([A-Z],?\s*){10,}', text)

    # "0.03962 is 0.03962" 같은 동어반복
    tautology = re.findall(r'(\d+\.?\d*)\s+is\s+\1', text)

    # "Yes, 28*20=560" 반복
    equation_repeat = re.findall(r'(Yes,\s*\d+\*\d+=\d+)', text)

    patterns['number_repeat'] = len(number_repeat)
    patterns['char_repeat'] = len(char_repeat)
    patterns['tautology'] = len(tautology)
    patterns['equation_repeat'] = len(equation_repeat)

    return patterns

def detect_context_drift(text_3bit, problem):
    """문맥 이탈 감지 - 완전히 다른 문제를 풀고 있는지"""
    # 마지막 500자에서 원래 문제와 관련 없는 내용인지 확인
    last_part = text_3bit[-1000:].lower()

    # 원래 문제의 핵심 키워드 추출
    problem_lower = problem.lower()

    # 명백한 문맥 이탈 패턴
    drift_patterns = [
        # f(-2) 문제에서 "sum of three terms each being 1/2" 푸는 경우
        (r'sum of three terms', r'f\('),
        # 다른 문제를 푸는 명백한 신호
        (r'the problem asks for', None),
    ]

    # 마지막 부분에서 원래 문제와 전혀 다른 계산을 하는지
    if 'f(x)' in problem_lower or 'f(-' in problem_lower:
        if 'sum of three terms' in last_part or 'each being' in last_part:
            return True, "완전히 다른 문제 풀이 (함수값→분수합)"

    return False, None

def analyze_error_type(d3, d4, problem, gold_answer):
    """개별 케이스의 오류 유형 분석"""

    text_3bit = d3['generated_text']
    text_4bit = d4['generated_text']

    answer_3bit = extract_boxed_answer(text_3bit)
    answer_4bit = extract_boxed_answer(text_4bit)

    words_3bit = len(text_3bit.split())
    words_4bit = len(text_4bit.split())

    rep_patterns = count_repetition_patterns(text_3bit)

    # 토큰 다양성
    tokens = text_3bit.split()
    diversity = len(set(tokens)) / len(tokens) if tokens else 0

    analysis = {
        'answer_3bit': answer_3bit,
        'answer_4bit': answer_4bit,
        'gold_answer': gold_answer,
        'words_3bit': words_3bit,
        'words_4bit': words_4bit,
        'diversity': diversity,
        'rep_patterns': rep_patterns,
        'primary_error': None,
        'error_category': None,
        'error_subcategory': None,
        'error_description': None,
        'evidence': [],
    }

    # ===== 1단계: 반복/붕괴 감지 (E) =====
    # 이것이 가장 명확하므로 먼저 체크

    # E2: 토큰/숫자 반복 붕괴 (다양성 < 0.05)
    if diversity < 0.05:
        analysis['primary_error'] = 'E2'
        analysis['error_category'] = 'E. 반복 및 붕괴'
        analysis['error_subcategory'] = 'E2. 토큰/숫자/문자 반복 붕괴'
        analysis['error_description'] = f'토큰 다양성 {diversity:.3f} (극심한 반복)'
        analysis['evidence'].append(f'다양성: {diversity:.4f}')
        if rep_patterns['number_repeat'] > 0:
            analysis['evidence'].append('숫자 연속 반복 감지')
        if rep_patterns['char_repeat'] > 0:
            analysis['evidence'].append('문자 연속 반복 감지')
        return analysis

    # E1: Wait/Perhaps 무한반복 (100회 이상)
    if rep_patterns['wait_count'] > 100 or rep_patterns['perhaps_count'] > 50:
        # 하지만 답변을 생성했다면 다른 오류 가능성 있음
        if not answer_3bit:
            analysis['primary_error'] = 'E1'
            analysis['error_category'] = 'E. 반복 및 붕괴'
            analysis['error_subcategory'] = 'E1. Wait/Perhaps 무한반복'
            analysis['error_description'] = f'Wait {rep_patterns["wait_count"]}회, Perhaps {rep_patterns["perhaps_count"]}회'
            analysis['evidence'].append(f'Wait: {rep_patterns["wait_count"]}회')
            if rep_patterns['equation_repeat'] > 5:
                analysis['evidence'].append(f'동일 수식 반복: {rep_patterns["equation_repeat"]}회')
            return analysis

    # E3: 문맥 이탈
    is_drift, drift_reason = detect_context_drift(text_3bit, problem)
    if is_drift:
        analysis['primary_error'] = 'E3'
        analysis['error_category'] = 'E. 반복 및 붕괴'
        analysis['error_subcategory'] = 'E3. 문맥 이탈'
        analysis['error_description'] = drift_reason
        analysis['evidence'].append('원래 문제와 무관한 내용으로 응답 종료')
        return analysis

    # ===== 2단계: 답변이 있는 경우 오류 분석 =====
    if answer_3bit and answer_3bit != gold_answer:
        # 응답 내용 분석 필요

        # 4-bit 응답과 비교하여 어디서 갈라졌는지 확인
        # 시작 부분이 비슷한지?
        start_3bit = text_3bit[:500].lower()
        start_4bit = text_4bit[:500].lower()

        # D1: 논리적 추론 오류 - 올바른 계산 후 잘못된 재검증
        # "However", "But wait", "upon verifying" 후 잘못된 결론
        reconsider_patterns = [
            r'however,?\s+upon\s+verif',
            r'but\s+wait',
            r'let\s+me\s+reconsider',
            r'wait,?\s+that\s+(doesn\'t|can\'t|isn\'t)',
            r'hmm,?\s+that\s+seems\s+wrong',
        ]

        for pattern in reconsider_patterns:
            if re.search(pattern, text_3bit.lower()):
                # 재검증 후 결론이 바뀌었는지 확인
                analysis['primary_error'] = 'D1'
                analysis['error_category'] = 'D. 추론 오류'
                analysis['error_subcategory'] = 'D1. 논리적 추론 오류'
                analysis['error_description'] = '올바른 풀이 후 불필요한 재검증으로 잘못된 결론 도출'
                analysis['evidence'].append('재검증 패턴 감지')
                return analysis

        # A2: 맥락적 간과 - 조건 무시
        # 예: "2는 유일한 짝수 소수" 인식했지만 제외하지 않음
        if 'only even prime' in text_3bit.lower() or '2 is the only' in text_3bit.lower():
            if 'odd' in problem.lower() and answer_3bit != answer_4bit:
                # 조건을 인식했지만 적용하지 않음
                analysis['primary_error'] = 'A2'
                analysis['error_category'] = 'A. 개념 오류'
                analysis['error_subcategory'] = 'A2. 맥락적 간과'
                analysis['error_description'] = '조건을 인식했으나 결론에 반영하지 않음'
                analysis['evidence'].append('짝수/홀수 조건 인식했으나 무시')
                return analysis

        # C1: 순수 계산 오류 - 산술 실수
        # 간단한 사칙연산 오류인지 확인
        # 예: 36-19=17인데 19라고 답한 경우
        arithmetic_error_indicators = [
            r'\d+\s*[-+*/]\s*\d+\s*=\s*\d+',  # 기본 연산
        ]

        # 마지막 계산 부분에서 오류가 있는지
        # 이것은 휴리스틱으로 판단하기 어려우므로 기본적으로 C1으로 분류

        # B2: 공식 오류 - 잘못된 공식 적용
        if 'formula' in text_3bit.lower() or 'theorem' in text_3bit.lower():
            # 공식을 언급했지만 결과가 틀림
            pass

        # 기본값: 실행 오류 (계산 오류)
        analysis['primary_error'] = 'C1'
        analysis['error_category'] = 'C. 실행 오류'
        analysis['error_subcategory'] = 'C1. 계산 오류'
        analysis['error_description'] = '계산 과정에서 산술적 실수'
        analysis['evidence'].append(f'3-bit 답: {answer_3bit}, 정답: {gold_answer}')
        return analysis

    # ===== 3단계: 답변 미생성 =====
    if not answer_3bit:
        # Wait 반복이 많으면 E1
        if rep_patterns['wait_count'] > 50:
            analysis['primary_error'] = 'E1'
            analysis['error_category'] = 'E. 반복 및 붕괴'
            analysis['error_subcategory'] = 'E1. Wait/Perhaps 무한반복'
            analysis['error_description'] = f'Wait {rep_patterns["wait_count"]}회 반복 후 답변 미생성'
            analysis['evidence'].append(f'Wait: {rep_patterns["wait_count"]}회')
            return analysis

        # 다양성이 낮으면 E2
        if diversity < 0.15:
            analysis['primary_error'] = 'E2'
            analysis['error_category'] = 'E. 반복 및 붕괴'
            analysis['error_subcategory'] = 'E2. 토큰/숫자/문자 반복 붕괴'
            analysis['error_description'] = f'토큰 다양성 {diversity:.3f}로 답변 생성 실패'
            analysis['evidence'].append(f'다양성: {diversity:.4f}')
            return analysis

        # 그 외 - 응답 중단
        analysis['primary_error'] = 'E1'
        analysis['error_category'] = 'E. 반복 및 붕괴'
        analysis['error_subcategory'] = 'E1. Wait/Perhaps 무한반복'
        analysis['error_description'] = '추론 중 반복/중단으로 답변 미생성'
        return analysis

    return analysis

def main():
    print("="*70)
    print("GPTQ 3-bit 오류 재분류 (Quantization Meets Reasoning 논문 체계)")
    print("="*70)

    # 데이터 로드
    data_3bit = load_json(GPTQ_3BIT_PATH)
    data_4bit = load_json(GPTQ_4BIT_PATH)

    # 실패 케이스 수집
    failures = []

    for i, (d3, d4) in enumerate(zip(data_3bit, data_4bit)):
        correct_3bit = d3['metrics'].get('extractive_match', 0) == 1.0
        correct_4bit = d4['metrics'].get('extractive_match', 0) == 1.0

        if correct_4bit and not correct_3bit:
            problem = extract_problem(d3['full_prompt'])
            gold_raw = d3['gold']
            gold = gold_raw[0] if isinstance(gold_raw, list) else gold_raw
            gold_answer = extract_boxed_answer(gold)

            analysis = analyze_error_type(d3, d4, problem, gold_answer)
            analysis['index'] = i
            analysis['problem'] = problem
            analysis['text_3bit'] = d3['generated_text']
            analysis['text_4bit'] = d4['generated_text']

            failures.append(analysis)

    print(f"\n총 실패 케이스: {len(failures)}개\n")

    # 카테고리별 집계 (None 처리)
    category_counts = Counter([f['error_category'] or 'Unknown' for f in failures])
    subcategory_counts = Counter([f['error_subcategory'] or 'Unknown' for f in failures])

    print("="*70)
    print("오류 카테고리별 분포")
    print("="*70)

    for cat, count in sorted(category_counts.items(), key=lambda x: x[0]):
        pct = count / len(failures) * 100
        bar = '█' * int(pct / 2)
        print(f"{cat}: {count}개 ({pct:.1f}%) {bar}")

    print("\n" + "="*70)
    print("오류 세부 유형별 분포")
    print("="*70)

    for subcat, count in sorted(subcategory_counts.items(), key=lambda x: x[0]):
        pct = count / len(failures) * 100
        bar = '█' * int(pct / 2)
        print(f"{subcat}: {count}개 ({pct:.1f}%) {bar}")

    # 각 카테고리별 대표 예시 추출
    examples_by_category = defaultdict(list)
    for f in failures:
        subcat = f['error_subcategory'] or 'Unknown'
        examples_by_category[subcat].append(f)

    # 결과 저장
    output = {
        'summary': {
            'total_failures': len(failures),
            'category_counts': dict(category_counts),
            'subcategory_counts': dict(subcategory_counts),
        },
        'failures': []
    }

    for f in failures:
        output['failures'].append({
            'index': f['index'],
            'problem': f['problem'][:300],
            'gold_answer': f['gold_answer'],
            'answer_3bit': f['answer_3bit'],
            'answer_4bit': f['answer_4bit'],
            'words_3bit': f['words_3bit'],
            'words_4bit': f['words_4bit'],
            'diversity': round(f['diversity'], 4),
            'rep_patterns': f['rep_patterns'],
            'primary_error': f['primary_error'],
            'error_category': f['error_category'],
            'error_subcategory': f['error_subcategory'],
            'error_description': f['error_description'],
            'evidence': f['evidence'],
            'text_3bit_start': f['text_3bit'][:600],
            'text_3bit_end': f['text_3bit'][-600:],
            'text_4bit_end': f['text_4bit'][-400:],
        })

    with open(f'{OUTPUT_DIR}/error_reclassification.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {OUTPUT_DIR}/error_reclassification.json")

    # 각 유형별 대표 예시 2개씩 출력
    print("\n" + "="*70)
    print("오류 유형별 대표 예시")
    print("="*70)

    for subcat in sorted(examples_by_category.keys()):
        cases = examples_by_category[subcat][:2]
        print(f"\n### {subcat} ({len(examples_by_category[subcat])}개)")

        for i, c in enumerate(cases):
            print(f"\n  예시 {i+1} (Index {c['index']}):")
            print(f"  문제: {c['problem'][:80]}...")
            print(f"  정답: {c['gold_answer']} | 3-bit: {c['answer_3bit']} | 4-bit: {c['answer_4bit']}")
            print(f"  설명: {c['error_description']}")
            print(f"  근거: {', '.join(c['evidence'])}")

if __name__ == "__main__":
    main()

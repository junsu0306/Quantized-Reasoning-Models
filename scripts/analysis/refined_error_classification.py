#!/usr/bin/env python3
"""
정밀한 오류 분류 - 실제 응답 내용을 면밀히 분석

오류 분류 우선순위:
1. E. 반복 및 붕괴 (가장 명확)
2. A. 개념 오류
3. D. 추론 오류
4. B. 방법 오류
5. C. 실행 오류
"""

import json
import re
from collections import Counter, defaultdict

BASE_DIR = "/Users/junsu/Projects/Quantized-Reasoning-Models"
OUTPUT_DIR = f"{BASE_DIR}/reports/gptq_3bit_error_analysis"

def load_reclassification():
    with open(f'{OUTPUT_DIR}/error_reclassification.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_arithmetic_error(text_3bit, answer_3bit, gold_answer):
    """실제 산술 계산 오류 감지"""
    # 마지막 계산 부분에서 잘못된 산술이 있는지
    # 예: "8 + 8 + 15 = 21" (실제는 31)

    # 패턴: X + Y + Z = W 형태의 계산
    calc_patterns = [
        r'(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)',
        r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)',
        r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)',
    ]

    errors = []
    for pattern in calc_patterns:
        matches = re.findall(pattern, text_3bit)
        for match in matches:
            if len(match) == 4:  # X + Y + Z = W
                a, b, c, result = map(int, match)
                if a + b + c != result:
                    errors.append(f'{a}+{b}+{c}={result} (실제: {a+b+c})')
            elif len(match) == 3:  # X - Y = Z 또는 X * Y = Z
                if '-' in pattern:
                    a, b, result = map(int, match)
                    if a - b != result:
                        errors.append(f'{a}-{b}={result} (실제: {a-b})')
                elif '*' in pattern:
                    a, b, result = map(int, match)
                    if a * b != result:
                        errors.append(f'{a}*{b}={result} (실제: {a*b})')

    return errors

def detect_context_drift(text_3bit, problem):
    """문맥 이탈 감지 - 완전히 다른 문제를 풀고 있는지"""
    # 마지막 500자에서 원래 문제와 전혀 다른 내용인지
    last_part = text_3bit[-1500:].lower()
    problem_lower = problem.lower()

    # 함수값 문제인데 분수 합을 계산하는 경우
    if 'f(x)' in problem_lower or 'f(-' in problem_lower:
        if 'sum of three terms' in last_part or 'each being 1/2' in last_part:
            return True, "완전히 다른 문제 풀이 (함수값→분수합)"
        if '1/2 + 1/2 + 1/2' in last_part:
            return True, "완전히 다른 문제 풀이 (분수 합)"

    # 조합론 문제인데 완전히 다른 계산
    if 'subcommittee' in problem_lower or 'committee' in problem_lower:
        if 'Yes, 28*20=560' in text_3bit[-2000:]:
            return True, "무의미한 수식 반복으로 문맥 상실"

    return False, None

def refined_classify(failure):
    """정제된 오류 분류"""

    text_3bit = failure.get('text_3bit_start', '') + failure.get('text_3bit_end', '')
    problem = failure['problem']
    answer_3bit = failure['answer_3bit']
    gold_answer = failure['gold_answer']
    diversity = failure['diversity']
    rep = failure['rep_patterns']

    # 1. E2: 토큰 다양성 붕괴 (다양성 < 0.05)
    if diversity < 0.05:
        return 'E2', 'E. 반복 및 붕괴', 'E2. 토큰/숫자/문자 반복 붕괴', f'다양성 {diversity:.3f}'

    # 2. E3: 문맥 이탈
    is_drift, drift_reason = detect_context_drift(failure.get('text_3bit_end', ''), problem)
    if is_drift:
        return 'E3', 'E. 반복 및 붕괴', 'E3. 문맥 이탈', drift_reason

    # 3. E1: Wait 무한반복 (답변 미생성)
    if not answer_3bit and (rep['wait_count'] > 50 or diversity < 0.1):
        return 'E1', 'E. 반복 및 붕괴', 'E1. Wait/Perhaps 무한반복', f'Wait {rep["wait_count"]}회'

    # 4. 답변이 있는 경우 - 상세 분석
    if answer_3bit:
        # C1: 실제 산술 계산 오류 감지
        arith_errors = detect_arithmetic_error(failure.get('text_3bit_end', ''), answer_3bit, gold_answer)
        if arith_errors:
            return 'C1', 'C. 실행 오류', 'C1. 계산 오류', f'산술 오류: {arith_errors[0]}'

        # A2: 맥락적 간과 - 조건 인식했으나 무시
        if 'only even prime' in text_3bit.lower() or '2 is the only' in text_3bit.lower():
            if 'odd' in problem.lower():
                return 'A2', 'A. 개념 오류', 'A2. 맥락적 간과', '짝수 조건 인식했으나 미적용'

        # B2: 공식/규칙 오류 - 잘못된 공식 적용
        # (추가 분석 필요)

        # D1: 논리적 추론 오류 - 올바른 풀이 후 잘못된 재검증
        # 재검증 패턴이 있고, 그 후에 결론이 바뀐 경우
        reconsider_patterns = [
            (r'however,?\s+upon\s+verif', 'However upon verifying'),
            (r'wait,?\s+that\s+(doesn\'t|can\'t)', 'Wait that doesn\'t'),
            (r'let\s+me\s+reconsider', 'Let me reconsider'),
            (r'hmm,?\s+that\s+seems\s+wrong', 'Hmm that seems wrong'),
            (r'but\s+wait', 'But wait'),
        ]

        for pattern, desc in reconsider_patterns:
            if re.search(pattern, text_3bit.lower()):
                # 재검증 후 답이 바뀌었는지 확인 (간단히 판단)
                return 'D1', 'D. 추론 오류', 'D1. 논리적 추론 오류', f'재검증 패턴: {desc}'

        # 기본: C1 계산 오류
        return 'C1', 'C. 실행 오류', 'C1. 계산 오류', f'답변 {answer_3bit} ≠ {gold_answer}'

    # 5. 답변 미생성 - 다양성 체크
    if diversity < 0.15:
        return 'E2', 'E. 반복 및 붕괴', 'E2. 토큰/숫자/문자 반복 붕괴', f'다양성 {diversity:.3f}'

    return 'E1', 'E. 반복 및 붕괴', 'E1. Wait/Perhaps 무한반복', '추론 중단'

def main():
    data = load_reclassification()
    failures = data['failures']

    print("="*70)
    print("정밀 오류 재분류")
    print("="*70)

    refined_results = []

    for f in failures:
        code, cat, subcat, desc = refined_classify(f)
        f['refined_error'] = code
        f['refined_category'] = cat
        f['refined_subcategory'] = subcat
        f['refined_description'] = desc
        refined_results.append(f)

    # 집계
    cat_counts = Counter([f['refined_category'] for f in refined_results])
    subcat_counts = Counter([f['refined_subcategory'] for f in refined_results])

    print("\n[정밀 분류 결과]")
    print("-"*50)

    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        pct = count / len(refined_results) * 100
        bar = '█' * int(pct / 2)
        print(f"{cat}: {count}개 ({pct:.1f}%) {bar}")

    print("\n[세부 유형]")
    print("-"*50)

    for subcat in sorted(subcat_counts.keys()):
        count = subcat_counts[subcat]
        pct = count / len(refined_results) * 100
        bar = '█' * int(pct / 2)
        print(f"{subcat}: {count}개 ({pct:.1f}%) {bar}")

    # 결과 저장
    output = {
        'summary': {
            'total': len(refined_results),
            'category_counts': dict(cat_counts),
            'subcategory_counts': dict(subcat_counts),
        },
        'failures': refined_results
    }

    with open(f'{OUTPUT_DIR}/refined_classification.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {OUTPUT_DIR}/refined_classification.json")

    # 유형별 예시 출력
    print("\n" + "="*70)
    print("유형별 대표 예시")
    print("="*70)

    by_subcat = defaultdict(list)
    for f in refined_results:
        by_subcat[f['refined_subcategory']].append(f)

    for subcat in sorted(by_subcat.keys()):
        cases = by_subcat[subcat][:2]
        print(f"\n### {subcat} ({len(by_subcat[subcat])}개)")

        for i, c in enumerate(cases):
            print(f"\n예시 {i+1} (Index {c['index']}):")
            print(f"  문제: {c['problem'][:60]}...")
            print(f"  정답: {c['gold_answer']} | 3-bit: {c['answer_3bit']}")
            print(f"  분류: {c['refined_description']}")
            if c.get('text_3bit_end'):
                end_text = c['text_3bit_end'][-200:].replace('\n', ' ')
                print(f"  응답 끝: ...{end_text}")

if __name__ == "__main__":
    main()

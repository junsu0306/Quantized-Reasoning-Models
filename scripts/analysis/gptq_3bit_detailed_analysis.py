#!/usr/bin/env python3
"""GPTQ 3-bit vs 4-bit 상세 분석 및 시각화"""

import json
import re
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로
BASE_DIR = "/Users/junsu/Projects/Quantized-Reasoning-Models"
GPTQ_3BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1-seed42/MATH-500.jsonl"
GPTQ_4BIT_PATH = f"{BASE_DIR}/outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1-seed42/MATH-500.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/reports/gptq_3bit_error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_jsonl(path):
    """JSON 또는 JSONL 파일 로드"""
    with open(path, 'r') as f:
        content = f.read().strip()

    # JSON 배열인지 JSONL인지 확인
    if content.startswith('['):
        return json.loads(content)
    else:
        # JSONL 형식
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data

def count_markers(text):
    """반복 마커 카운트"""
    markers = {
        'Wait': len(re.findall(r'\bWait\b', text)),
        'Perhaps': len(re.findall(r'\bPerhaps\b', text)),
        'Let me': len(re.findall(r'Let me', text)),
        'Hmm': len(re.findall(r'\bHmm\b', text)),
        'So': len(re.findall(r'\bSo,?\b', text)),
        'But': len(re.findall(r'\bBut\b', text)),
    }
    return markers

def analyze_response(text):
    """응답 분석"""
    word_count = len(text.split())
    char_count = len(text)
    markers = count_markers(text)
    has_answer = bool(re.search(r'\\boxed\{', text))

    # 토큰 다양성 (unique words / total words)
    words = text.split()
    unique_words = set(words)
    diversity = len(unique_words) / len(words) if words else 0

    return {
        'word_count': word_count,
        'char_count': char_count,
        'markers': markers,
        'total_markers': sum(markers.values()),
        'has_answer': has_answer,
        'diversity': diversity
    }

def extract_problem_text(full_prompt):
    """full_prompt에서 실제 문제 텍스트만 추출"""
    import re
    # User 태그와 "Please reason" 사이의 텍스트 추출
    match = re.search(r'User[｜\|]?>(.+?)(?:Please reason|$)', full_prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return full_prompt

def classify_problem_type(full_prompt):
    """문제 유형 분류"""
    # full_prompt에서 실제 문제만 추출
    problem = extract_problem_text(full_prompt)
    text = problem.lower()

    # 삼각함수: sin, cos, tan 등이 문제에 직접 포함된 경우
    if any(kw in text for kw in ['\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
                                  'sine', 'cosine', 'tangent', 'trigonometric']):
        return 'Trigonometry'
    elif any(kw in text for kw in ['probability', 'randomly', 'chance', 'expected value', 'dice', 'coin']):
        return 'Probability'
    elif any(kw in text for kw in ['integer', 'divisor', 'divisible', 'prime', 'gcd', 'lcm', 'modulo', 'remainder', 'factor']):
        return 'Number Theory'
    elif any(kw in text for kw in ['polygon', 'circle', 'area', 'perimeter', 'volume', 'rectangle', 'square', 'triangle', 'radius', 'diameter']):
        return 'Geometry'
    elif any(kw in text for kw in ['combination', 'permutation', 'how many ways', 'arrangements', 'choose', 'select']):
        return 'Combinatorics'
    elif any(kw in text for kw in ['equation', 'polynomial', 'solve for', 'find the value', 'simplify', 'evaluate', 'expression']):
        return 'Algebra'
    else:
        return 'Other'

def classify_loop_type(markers):
    """루프 유형 분류"""
    wait = markers.get('Wait', 0)
    perhaps = markers.get('Perhaps', 0)
    total = sum(markers.values())

    if wait > 1000:
        return 'severe_wait_loop'
    elif wait > 100:
        return 'moderate_wait_loop'
    elif perhaps > 50:
        return 'perhaps_loop'
    elif total > 200:
        return 'mixed_loop'
    else:
        return 'no_loop'

def categorize_failure_reason(analysis_3bit, analysis_4bit):
    """실패 원인 분류"""
    reasons = []

    # 1. Wait 루프
    if analysis_3bit['markers']['Wait'] > 100:
        reasons.append('Wait 무한반복')

    # 2. 응답 길이 폭발
    ratio = analysis_3bit['word_count'] / max(analysis_4bit['word_count'], 1)
    if ratio > 5:
        reasons.append('응답길이 폭발')
    elif ratio > 3:
        reasons.append('응답길이 증가')

    # 3. 토큰 다양성 붕괴
    if analysis_3bit['diversity'] < 0.1:
        reasons.append('토큰다양성 붕괴')
    elif analysis_3bit['diversity'] < 0.15:
        reasons.append('토큰다양성 저하')

    # 4. 답변 생성 실패
    if not analysis_3bit['has_answer']:
        reasons.append('답변 미생성')

    # 5. 계산 오류 (답변은 있지만 틀림)
    if analysis_3bit['has_answer'] and analysis_4bit['has_answer']:
        reasons.append('계산 오류')

    return reasons if reasons else ['기타']

def main():
    print("GPTQ 3-bit vs 4-bit 상세 분석")
    print("="*60)

    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_3bit = load_jsonl(GPTQ_3BIT_PATH)
    data_4bit = load_jsonl(GPTQ_4BIT_PATH)

    print(f"   3-bit 데이터: {len(data_3bit)}개")
    print(f"   4-bit 데이터: {len(data_4bit)}개")

    # 문제별 매칭 (problem 또는 prompt 필드 기준)
    print("\n2. 결과 비교 중...")

    # 결과 분류
    only_4bit_correct = []  # 4bit만 맞춤 (3bit 실패)
    both_correct = []
    both_wrong = []
    only_3bit_correct = []

    for i, (d3, d4) in enumerate(zip(data_3bit, data_4bit)):
        # 정답 여부 확인 (extractive_match 또는 correct 필드)
        if 'metrics' in d3:
            correct_3bit = d3['metrics'].get('extractive_match', 0) == 1.0
            correct_4bit = d4['metrics'].get('extractive_match', 0) == 1.0
        else:
            correct_3bit = d3.get('correct', False)
            correct_4bit = d4.get('correct', False)

        text_3bit = d3.get('generated_text', '')
        text_4bit = d4.get('generated_text', '')
        problem = d3.get('full_prompt', d3.get('problem', d3.get('prompt', '')))
        gold_raw = d3.get('gold', d3.get('gold_answer', d3.get('answer', '')))
        gold_answer = gold_raw[0] if isinstance(gold_raw, list) else gold_raw

        analysis_3bit = analyze_response(text_3bit)
        analysis_4bit = analyze_response(text_4bit)

        record = {
            'index': i,
            'problem': problem,
            'gold_answer': gold_answer,
            'correct_3bit': correct_3bit,
            'correct_4bit': correct_4bit,
            'text_3bit': text_3bit,
            'text_4bit': text_4bit,
            'analysis_3bit': analysis_3bit,
            'analysis_4bit': analysis_4bit,
            'problem_type': classify_problem_type(problem),
            'loop_type': classify_loop_type(analysis_3bit['markers']),
        }

        if correct_4bit and not correct_3bit:
            record['failure_reasons'] = categorize_failure_reason(analysis_3bit, analysis_4bit)
            only_4bit_correct.append(record)
        elif correct_3bit and correct_4bit:
            both_correct.append(record)
        elif not correct_3bit and not correct_4bit:
            both_wrong.append(record)
        else:
            only_3bit_correct.append(record)

    print(f"   4-bit만 정답: {len(only_4bit_correct)}개 (3-bit 실패)")
    print(f"   둘 다 정답:  {len(both_correct)}개")
    print(f"   둘 다 오답:  {len(both_wrong)}개")
    print(f"   3-bit만 정답: {len(only_3bit_correct)}개")

    # 3. 실패 케이스 상세 분석
    print("\n3. 3-bit 실패 케이스 분석")
    print("-"*60)

    # 문제 유형별 분포
    type_dist = Counter([r['problem_type'] for r in only_4bit_correct])
    print("\n   [문제 유형별 분포]")
    for ptype, count in type_dist.most_common():
        pct = count / len(only_4bit_correct) * 100
        print(f"   - {ptype}: {count}개 ({pct:.1f}%)")

    # 루프 유형별 분포
    loop_dist = Counter([r['loop_type'] for r in only_4bit_correct])
    print("\n   [루프 유형별 분포]")
    for ltype, count in loop_dist.most_common():
        pct = count / len(only_4bit_correct) * 100
        print(f"   - {ltype}: {count}개 ({pct:.1f}%)")

    # 실패 원인별 분포
    reason_counter = Counter()
    for r in only_4bit_correct:
        for reason in r['failure_reasons']:
            reason_counter[reason] += 1

    print("\n   [실패 원인별 분포 (중복 가능)]")
    for reason, count in reason_counter.most_common():
        pct = count / len(only_4bit_correct) * 100
        print(f"   - {reason}: {count}개 ({pct:.1f}%)")

    # 통계
    avg_word_3bit = np.mean([r['analysis_3bit']['word_count'] for r in only_4bit_correct])
    avg_word_4bit = np.mean([r['analysis_4bit']['word_count'] for r in only_4bit_correct])
    avg_wait_3bit = np.mean([r['analysis_3bit']['markers']['Wait'] for r in only_4bit_correct])
    avg_wait_4bit = np.mean([r['analysis_4bit']['markers']['Wait'] for r in only_4bit_correct])
    avg_div_3bit = np.mean([r['analysis_3bit']['diversity'] for r in only_4bit_correct])
    avg_div_4bit = np.mean([r['analysis_4bit']['diversity'] for r in only_4bit_correct])

    print("\n   [통계 비교]")
    print(f"   평균 단어수: 3-bit={avg_word_3bit:.0f}, 4-bit={avg_word_4bit:.0f} ({avg_word_3bit/avg_word_4bit:.1f}배)")
    print(f"   평균 Wait:  3-bit={avg_wait_3bit:.0f}, 4-bit={avg_wait_4bit:.0f} ({avg_wait_3bit/max(avg_wait_4bit,1):.1f}배)")
    print(f"   평균 다양성: 3-bit={avg_div_3bit:.3f}, 4-bit={avg_div_4bit:.3f}")

    # ===== 시각화 =====
    print("\n4. 시각화 생성 중...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 4-1. 전체 결과 분류 파이차트
    ax = axes[0, 0]
    labels = ['4-bit만 정답\n(3-bit 실패)', '둘 다 정답', '둘 다 오답', '3-bit만 정답']
    sizes = [len(only_4bit_correct), len(both_correct), len(both_wrong), len(only_3bit_correct)]
    colors = ['#e74c3c', '#27ae60', '#95a5a6', '#3498db']
    explode = (0.05, 0, 0, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('GPTQ 3-bit vs 4-bit 결과 분포\n(MATH-500)', fontsize=14, fontweight='bold')

    # 4-2. 문제 유형별 실패 분포
    ax = axes[0, 1]
    types = list(type_dist.keys())
    counts = list(type_dist.values())
    colors_bar = plt.cm.Set3(np.linspace(0, 1, len(types)))
    bars = ax.barh(types, counts, color=colors_bar)
    ax.set_xlabel('실패 개수')
    ax.set_title('문제 유형별 3-bit 실패 분포', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=10)

    # 4-3. 루프 유형별 분포
    ax = axes[0, 2]
    loop_types = ['no_loop', 'moderate_wait_loop', 'severe_wait_loop', 'perhaps_loop', 'mixed_loop']
    loop_counts = [loop_dist.get(lt, 0) for lt in loop_types]
    colors_loop = ['#27ae60', '#f39c12', '#e74c3c', '#9b59b6', '#34495e']
    bars = ax.bar(range(len(loop_types)), loop_counts, color=colors_loop)
    ax.set_xticks(range(len(loop_types)))
    ax.set_xticklabels(['정상', '중간 Wait', '심각 Wait', 'Perhaps', '혼합'], rotation=45, ha='right')
    ax.set_ylabel('실패 개수')
    ax.set_title('루프 유형별 분포', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, loop_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', fontsize=10)

    # 4-4. 실패 원인별 분포 (수평 바)
    ax = axes[1, 0]
    reasons = list(reason_counter.keys())
    reason_counts = list(reason_counter.values())
    colors_reason = plt.cm.Reds(np.linspace(0.3, 0.9, len(reasons)))
    bars = ax.barh(reasons, reason_counts, color=colors_reason)
    ax.set_xlabel('실패 개수 (중복 가능)')
    ax.set_title('실패 원인별 분포', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, reason_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=10)

    # 4-5. 응답 길이 비교 (박스플롯)
    ax = axes[1, 1]
    word_3bit = [r['analysis_3bit']['word_count'] for r in only_4bit_correct]
    word_4bit = [r['analysis_4bit']['word_count'] for r in only_4bit_correct]
    bp = ax.boxplot([word_4bit, word_3bit], labels=['4-bit (정답)', '3-bit (오답)'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel('응답 길이 (단어)')
    ax.set_title('응답 길이 비교\n(3-bit 실패 케이스)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')

    # 4-6. Wait 마커 비교 (스캐터플롯)
    ax = axes[1, 2]
    wait_3bit = [r['analysis_3bit']['markers']['Wait'] for r in only_4bit_correct]
    wait_4bit = [r['analysis_4bit']['markers']['Wait'] for r in only_4bit_correct]
    ax.scatter(wait_4bit, wait_3bit, alpha=0.6, c='#e74c3c', s=50)
    ax.plot([0, max(wait_4bit)*1.1], [0, max(wait_4bit)*1.1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('4-bit Wait 횟수')
    ax.set_ylabel('3-bit Wait 횟수')
    ax.set_title('Wait 마커 비교\n(대각선 위 = 3-bit 더 많음)', fontsize=14, fontweight='bold')
    ax.set_xlim(-5, max(wait_4bit)*1.1)
    ax.set_ylim(-5, max(wait_3bit)*1.1)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gptq_3bit_failure_overview.png', dpi=150, bbox_inches='tight')
    print(f"   저장: {OUTPUT_DIR}/gptq_3bit_failure_overview.png")

    # ===== 상세 시각화 2: 개별 문제 분석 =====
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

    # 5-1. 토큰 다양성 비교
    ax = axes2[0, 0]
    div_3bit = [r['analysis_3bit']['diversity'] for r in only_4bit_correct]
    div_4bit = [r['analysis_4bit']['diversity'] for r in only_4bit_correct]
    ax.scatter(div_4bit, div_3bit, alpha=0.6, c='#9b59b6', s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('4-bit 토큰 다양성')
    ax.set_ylabel('3-bit 토큰 다양성')
    ax.set_title('토큰 다양성 비교\n(대각선 아래 = 3-bit 다양성 감소)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)

    # 5-2. 응답 길이 vs Wait 마커 (3-bit)
    ax = axes2[0, 1]
    colors_by_type = {'no_loop': '#27ae60', 'moderate_wait_loop': '#f39c12',
                      'severe_wait_loop': '#e74c3c', 'perhaps_loop': '#9b59b6', 'mixed_loop': '#34495e'}
    for r in only_4bit_correct:
        ax.scatter(r['analysis_3bit']['word_count'], r['analysis_3bit']['markers']['Wait'],
                   alpha=0.6, c=colors_by_type.get(r['loop_type'], 'gray'), s=50)

    handles = [mpatches.Patch(color=c, label=l) for l, c in colors_by_type.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlabel('응답 길이 (단어)')
    ax.set_ylabel('Wait 횟수')
    ax.set_title('3-bit 응답 길이 vs Wait 횟수\n(루프 유형별 색상)', fontsize=14, fontweight='bold')

    # 5-3. 문제 유형별 평균 Wait 횟수
    ax = axes2[1, 0]
    type_wait = defaultdict(list)
    for r in only_4bit_correct:
        type_wait[r['problem_type']].append(r['analysis_3bit']['markers']['Wait'])

    types_sorted = sorted(type_wait.keys(), key=lambda x: -np.mean(type_wait[x]))
    avg_waits = [np.mean(type_wait[t]) for t in types_sorted]
    bars = ax.bar(range(len(types_sorted)), avg_waits, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(types_sorted))))
    ax.set_xticks(range(len(types_sorted)))
    ax.set_xticklabels(types_sorted, rotation=45, ha='right')
    ax.set_ylabel('평균 Wait 횟수')
    ax.set_title('문제 유형별 평균 Wait 횟수\n(3-bit 실패 케이스)', fontsize=14, fontweight='bold')

    # 5-4. 길이 비율 히스토그램 (3-bit / 4-bit)
    ax = axes2[1, 1]
    length_ratios = [r['analysis_3bit']['word_count'] / max(r['analysis_4bit']['word_count'], 1)
                     for r in only_4bit_correct]
    ax.hist(length_ratios, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', label='1:1 비율')
    ax.axvline(x=np.median(length_ratios), color='orange', linestyle='-', label=f'중앙값: {np.median(length_ratios):.1f}')
    ax.set_xlabel('응답 길이 비율 (3-bit / 4-bit)')
    ax.set_ylabel('빈도')
    ax.set_title('응답 길이 비율 분포\n(>1이면 3-bit가 더 긺)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gptq_3bit_failure_details.png', dpi=150, bbox_inches='tight')
    print(f"   저장: {OUTPUT_DIR}/gptq_3bit_failure_details.png")

    # ===== 대표 실패 케이스 저장 =====
    print("\n5. 대표 실패 케이스 저장 중...")

    # 각 루프 유형별 대표 예시 추출
    examples = {}
    for loop_type in ['severe_wait_loop', 'moderate_wait_loop', 'no_loop', 'perhaps_loop']:
        cases = [r for r in only_4bit_correct if r['loop_type'] == loop_type]
        if cases:
            # 가장 극단적인 케이스 선택
            if loop_type == 'severe_wait_loop':
                case = max(cases, key=lambda x: x['analysis_3bit']['markers']['Wait'])
            elif loop_type == 'moderate_wait_loop':
                case = max(cases, key=lambda x: x['analysis_3bit']['markers']['Wait'])
            else:
                case = cases[0]

            examples[loop_type] = {
                'index': case['index'],
                'problem': case['problem'][:500] + '...' if len(case['problem']) > 500 else case['problem'],
                'gold_answer': case['gold_answer'],
                'problem_type': case['problem_type'],
                'failure_reasons': case['failure_reasons'],
                '3bit_stats': {
                    'word_count': case['analysis_3bit']['word_count'],
                    'wait_count': case['analysis_3bit']['markers']['Wait'],
                    'diversity': round(case['analysis_3bit']['diversity'], 4),
                    'has_answer': case['analysis_3bit']['has_answer'],
                    'first_500_chars': case['text_3bit'][:500],
                    'last_500_chars': case['text_3bit'][-500:] if len(case['text_3bit']) > 500 else case['text_3bit'],
                },
                '4bit_stats': {
                    'word_count': case['analysis_4bit']['word_count'],
                    'wait_count': case['analysis_4bit']['markers']['Wait'],
                    'diversity': round(case['analysis_4bit']['diversity'], 4),
                    'has_answer': case['analysis_4bit']['has_answer'],
                    'first_500_chars': case['text_4bit'][:500],
                    'last_500_chars': case['text_4bit'][-500:] if len(case['text_4bit']) > 500 else case['text_4bit'],
                }
            }

    with open(f'{OUTPUT_DIR}/representative_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"   저장: {OUTPUT_DIR}/representative_examples.json")

    # ===== 전체 실패 케이스 상세 CSV =====
    import csv
    with open(f'{OUTPUT_DIR}/all_failures.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'problem_type', 'loop_type', 'failure_reasons',
                         '3bit_words', '4bit_words', 'word_ratio',
                         '3bit_wait', '4bit_wait',
                         '3bit_diversity', '4bit_diversity',
                         '3bit_has_answer', '4bit_has_answer',
                         'gold_answer', 'problem_preview'])

        for r in only_4bit_correct:
            writer.writerow([
                r['index'],
                r['problem_type'],
                r['loop_type'],
                '|'.join(r['failure_reasons']),
                r['analysis_3bit']['word_count'],
                r['analysis_4bit']['word_count'],
                round(r['analysis_3bit']['word_count'] / max(r['analysis_4bit']['word_count'], 1), 2),
                r['analysis_3bit']['markers']['Wait'],
                r['analysis_4bit']['markers']['Wait'],
                round(r['analysis_3bit']['diversity'], 4),
                round(r['analysis_4bit']['diversity'], 4),
                r['analysis_3bit']['has_answer'],
                r['analysis_4bit']['has_answer'],
                r['gold_answer'],
                r['problem'][:200]
            ])
    print(f"   저장: {OUTPUT_DIR}/all_failures.csv")

    print("\n" + "="*60)
    print("분석 완료!")
    print(f"결과 저장 위치: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

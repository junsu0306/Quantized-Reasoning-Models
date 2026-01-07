#!/usr/bin/env python3
"""GPTQ 3-bit 추가 시각화: 실패 패턴 심층 분석"""

import json
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import csv

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = "/Users/junsu/Projects/Quantized-Reasoning-Models"
OUTPUT_DIR = f"{BASE_DIR}/reports/gptq_3bit_error_analysis"

def load_csv_data():
    """CSV 데이터 로드"""
    data = []
    with open(f'{OUTPUT_DIR}/all_failures.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def create_failure_pattern_viz():
    """실패 패턴 심층 시각화"""
    data = load_csv_data()

    fig = plt.figure(figsize=(20, 16))

    # 1. 실패 원인 조합 분석 (Upset-like plot)
    ax1 = fig.add_subplot(2, 2, 1)
    reason_combos = defaultdict(int)
    for row in data:
        reasons = tuple(sorted(row['failure_reasons'].split('|')))
        reason_combos[reasons] += 1

    # 상위 10개 조합
    top_combos = sorted(reason_combos.items(), key=lambda x: -x[1])[:10]
    combo_labels = ['\n'.join(c[0][:3]) + ('...' if len(c[0]) > 3 else '') for c in top_combos]
    combo_counts = [c[1] for c in top_combos]

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(combo_counts)))
    bars = ax1.barh(range(len(combo_labels)), combo_counts, color=colors)
    ax1.set_yticks(range(len(combo_labels)))
    ax1.set_yticklabels(combo_labels, fontsize=8)
    ax1.set_xlabel('빈도')
    ax1.set_title('실패 원인 조합 분포 (Top 10)', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, combo_counts):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{count}', va='center', fontsize=9)

    # 2. 문제 유형 × 루프 유형 히트맵
    ax2 = fig.add_subplot(2, 2, 2)
    problem_types = sorted(set(row['problem_type'] for row in data))
    loop_types = ['no_loop', 'moderate_wait_loop', 'severe_wait_loop', 'mixed_loop', 'perhaps_loop']

    heatmap_data = np.zeros((len(problem_types), len(loop_types)))
    for row in data:
        pt_idx = problem_types.index(row['problem_type'])
        lt = row['loop_type']
        if lt in loop_types:
            lt_idx = loop_types.index(lt)
            heatmap_data[pt_idx, lt_idx] += 1

    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(loop_types)))
    ax2.set_xticklabels(['정상', '중간 Wait', '심각 Wait', '혼합', 'Perhaps'], rotation=45, ha='right')
    ax2.set_yticks(range(len(problem_types)))
    ax2.set_yticklabels(problem_types)
    ax2.set_title('문제 유형 × 루프 유형 분포', fontsize=14, fontweight='bold')

    # 히트맵에 숫자 표시
    for i in range(len(problem_types)):
        for j in range(len(loop_types)):
            if heatmap_data[i, j] > 0:
                ax2.text(j, i, int(heatmap_data[i, j]), ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax2)

    # 3. 응답 길이 vs 토큰 다양성 (실패 원인별 색상)
    ax3 = fig.add_subplot(2, 2, 3)
    reason_colors = {
        'Wait 무한반복': '#e74c3c',
        '응답길이 폭발': '#f39c12',
        '토큰다양성 붕괴': '#9b59b6',
        '계산 오류': '#3498db',
        '답변 미생성': '#95a5a6'
    }

    for row in data:
        word_3bit = int(row['3bit_words'])
        div_3bit = float(row['3bit_diversity'])
        reasons = row['failure_reasons'].split('|')

        # 주요 원인 색상 선택
        color = '#34495e'
        for reason in reasons:
            if reason in reason_colors:
                color = reason_colors[reason]
                break

        ax3.scatter(word_3bit, div_3bit, c=color, alpha=0.6, s=60)

    # 범례
    handles = [mpatches.Patch(color=c, label=l) for l, c in reason_colors.items()]
    ax3.legend(handles=handles, loc='upper right', fontsize=8)
    ax3.set_xlabel('3-bit 응답 길이 (단어)')
    ax3.set_ylabel('3-bit 토큰 다양성')
    ax3.set_title('응답 길이 vs 토큰 다양성\n(주요 실패 원인별 색상)', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='다양성 임계값')

    # 4. 문제 유형별 성능 저하 정도 (Word Ratio 박스플롯)
    ax4 = fig.add_subplot(2, 2, 4)
    type_ratios = defaultdict(list)
    for row in data:
        pt = row['problem_type']
        ratio = float(row['word_ratio'])
        type_ratios[pt].append(ratio)

    # 중앙값 기준 정렬
    types_sorted = sorted(type_ratios.keys(), key=lambda x: -np.median(type_ratios[x]))
    box_data = [type_ratios[t] for t in types_sorted]

    bp = ax4.boxplot(box_data, patch_artist=True, tick_labels=types_sorted)
    colors_box = plt.cm.Oranges(np.linspace(0.3, 0.9, len(types_sorted)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax4.set_xticklabels(types_sorted, rotation=45, ha='right')
    ax4.set_ylabel('응답 길이 비율 (3-bit / 4-bit)')
    ax4.set_title('문제 유형별 응답 길이 폭발 정도', fontsize=14, fontweight='bold')
    ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=3, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gptq_3bit_failure_patterns.png', dpi=150, bbox_inches='tight')
    print(f"저장: {OUTPUT_DIR}/gptq_3bit_failure_patterns.png")

def create_example_comparison_viz():
    """대표 예시 비교 시각화"""
    with open(f'{OUTPUT_DIR}/representative_examples.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    loop_types = ['severe_wait_loop', 'moderate_wait_loop', 'no_loop']
    loop_names = {'severe_wait_loop': '심각한 Wait 루프', 'moderate_wait_loop': '중간 Wait 루프', 'no_loop': '루프 없음'}

    for idx, loop_type in enumerate(loop_types):
        if loop_type not in examples:
            continue

        ex = examples[loop_type]
        ax = axes[idx // 2, idx % 2]

        # 3-bit vs 4-bit 비교 막대그래프
        metrics = ['단어수', 'Wait횟수', '다양성×100']
        values_3bit = [
            ex['3bit_stats']['word_count'],
            ex['3bit_stats']['wait_count'],
            ex['3bit_stats']['diversity'] * 100
        ]
        values_4bit = [
            ex['4bit_stats']['word_count'],
            ex['4bit_stats']['wait_count'],
            ex['4bit_stats']['diversity'] * 100
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, values_4bit, width, label='4-bit (정답)', color='#27ae60')
        bars2 = ax.bar(x + width/2, values_3bit, width, label='3-bit (오답)', color='#e74c3c')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_yscale('symlog')
        ax.legend()
        ax.set_title(f'{loop_names[loop_type]}\n문제 유형: {ex["problem_type"]}', fontsize=12, fontweight='bold')

        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}' if height > 1 else f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}' if height > 1 else f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    # 4번째 subplot: 요약 텍스트
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
GPTQ 3-bit 실패 패턴 요약

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 전체 통계:
  • 3-bit 실패 케이스: 78개 (15.6%)
  • 평균 응답 길이 증가: 2.8배
  • 평균 Wait 반복 증가: 3.8배

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 주요 실패 원인:
  1. 계산 오류 (78.2%)
     → 답변은 생성하지만 값이 틀림

  2. 토큰 다양성 붕괴 (39.7%)
     → 같은 토큰이 반복됨

  3. Wait 무한반복 (28.2%)
     → "Wait, let me..." 패턴 반복

  4. 응답 길이 폭발 (26.9%)
     → 4-bit 대비 5배 이상 길어짐

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 특징적 실패 모드:
  • 51.3%는 루프 없이 단순 계산 오류
  • 28.2%는 중간 정도 Wait 반복
  • 1.3%만 심각한 Wait 루프 (AWQ와 대비)
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gptq_3bit_example_comparison.png', dpi=150, bbox_inches='tight')
    print(f"저장: {OUTPUT_DIR}/gptq_3bit_example_comparison.png")

def create_awq_vs_gptq_comparison():
    """AWQ vs GPTQ 3-bit 비교 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 데이터 (기존 분석에서 가져옴)
    methods = ['AWQ 3-bit', 'GPTQ 3-bit']

    # 1. 성능 저하 케이스 수
    ax = axes[0]
    cases = [164, 78]
    bars = ax.bar(methods, cases, color=['#e74c3c', '#3498db'])
    ax.set_ylabel('실패 케이스 수')
    ax.set_title('성능 저하 케이스\n(4-bit만 정답)', fontsize=14, fontweight='bold')
    for bar, case in zip(bars, cases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{case}개', ha='center', fontsize=12, fontweight='bold')

    # 2. Wait 마커 평균
    ax = axes[1]
    waits = [1277.5, 110.9]
    bars = ax.bar(methods, waits, color=['#e74c3c', '#3498db'])
    ax.set_ylabel('평균 Wait 횟수')
    ax.set_title('Wait 마커 반복\n(실패 케이스)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    for bar, wait in zip(bars, waits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{wait:.0f}회', ha='center', fontsize=12, fontweight='bold')

    # 3. 응답 길이
    ax = axes[2]
    lengths = [16314, 6554]
    bars = ax.bar(methods, lengths, color=['#e74c3c', '#3498db'])
    ax.set_ylabel('평균 응답 길이 (단어)')
    ax.set_title('응답 길이\n(실패 케이스)', fontsize=14, fontweight='bold')
    for bar, length in zip(bars, lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{length:,}', ha='center', fontsize=12, fontweight='bold')

    # 비율 표시
    for i, ax in enumerate(axes):
        ratio = [164/78, 1277.5/110.9, 16314/6554][i]
        ax.text(0.5, 0.02, f'AWQ/GPTQ = {ratio:.1f}배',
                transform=ax.transAxes, ha='center', fontsize=10,
                style='italic', color='gray')

    plt.suptitle('AWQ vs GPTQ 3-bit 비교 (MATH-500 기준)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/awq_vs_gptq_3bit_comparison.png', dpi=150, bbox_inches='tight')
    print(f"저장: {OUTPUT_DIR}/awq_vs_gptq_3bit_comparison.png")

def main():
    print("GPTQ 3-bit 추가 시각화 생성 중...")
    print("="*60)

    create_failure_pattern_viz()
    create_example_comparison_viz()
    create_awq_vs_gptq_comparison()

    print("\n" + "="*60)
    print("완료!")

if __name__ == "__main__":
    main()

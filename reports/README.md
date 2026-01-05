# Reports 폴더 구조 및 사용 가이드

본 폴더는 양자화 추론 모델(DeepSeek-R1-Distill-Qwen-1.5B)에 대한 종합 분석 결과를 포함합니다.

## 📁 파일 구조

```
reports/
├── README.md                                  # 본 파일
├── COMPREHENSIVE_QUANTIZATION_ANALYSIS.md     # 🏆 메인 종합 보고서 (논문용)
├── ANALYSIS_REPORT.md                         # MATH-500 상세 분석 (기존)
├── AWQ3_SUCCESS_FAILURE_REPORT.md            # AWQ 3-bit 성공/실패 요인 (기존)
├── tables/                                    # 논문용 표 (CSV, LaTeX, Markdown)
│   ├── overall_performance.*
│   ├── MATH_500_comparison.*
│   ├── AIME_90_comparison.*
│   ├── error_analysis.*
│   └── *_quantization_comparison.*
├── statistics/                                # 통계 데이터
│   └── summary.json
└── intermediate_data/                         # 중간 분석 데이터
    ├── *_MATH-500.{json,csv}
    └── *_AIME-90.{json,csv}
```

---

## 📊 주요 보고서

### 1. COMPREHENSIVE_QUANTIZATION_ANALYSIS.md (메인 보고서)

**🎯 논문 작성 시 이 보고서를 사용하세요!**

**포함 내용**:
- 6개 모델 비교 (Baseline, AWQ 3/4-bit, GPTQ 3/4-bit, KV-Quant* KV4)
- 2개 데이터셋 분석 (MATH-500, AIME-90)
- 양자화 방법별 상세 비교
- 에러 패턴 종합 분석
- Repetition Degeneration 메커니즘 규명
- 논문용 권장사항 및 결론

**핵심 결과**:
| 양자화 방법 | MATH-500 | AIME-90 | 추천도 |
|------------|---------|---------|--------|
| KV-Quant* KV4 | 84.20% (-0.8pp) | 20.00% | 🏆 최고 |
| AWQ 4-bit | 83.40% (-1.6pp) | 20.00% | ✅ 우수 |
| GPTQ 4-bit | 83.00% (-2.0pp) | 18.89% | ✅ 우수 |
| GPTQ 3-bit | 71.40% (-13.6pp) | 10.00% | ⚠️ 제한적 |
| AWQ 3-bit | 52.60% (-32.4pp) | 6.67% | ❌ 비권장 |

### 2. ANALYSIS_REPORT.md (MATH-500 심층 분석)

**포함 내용**:
- MATH-500 벤치마크 상세 분석
- AWQ 3-bit vs 4-bit vs Baseline 비교
- 문제 유형별 성능 (Trigonometry, Geometry, etc.)
- Repetition Degeneration 상세 분석
- 토큰 다양성, 응답 길이 등 품질 지표

**핵심 발견**:
- AWQ 3-bit는 삼각함수에서 63.6pp 하락
- 88.6% 케이스에서 답변 생성 실패
- 토큰 다양성 72% 감소

### 3. AWQ3_SUCCESS_FAILURE_REPORT.md (성공/실패 요인 분석)

**포함 내용**:
- AWQ 3-bit 52.6% 성공 vs 47.4% 실패 원인 규명
- 문제 복잡도별 성공률 (Baseline 응답 길이 기준)
- "Race Condition" 메커니즘 (답 도출 vs 반복 루프)
- 실패 예측 지표

**핵심 발견**:
- 간단한 문제 (< 1,500 단어): 70% 성공
- 복잡한 문제 (> 3,000 단어): 19% 성공
- Baseline 응답 > 3,000 단어 → 81% 실패 확률

---

## 📈 논문용 표 (tables/)

모든 표는 **3가지 형식**으로 제공:
- **`.csv`**: 데이터 처리, 추가 분석용
- **`.tex`**: LaTeX 논문 작성용
- **`.md`**: Markdown 문서, README용

### 주요 표 목록

1. **overall_performance**: 전체 모델 성능 비교
2. **MATH_500_comparison**: MATH-500 데이터셋 상세 비교
3. **AIME_90_comparison**: AIME-90 데이터셋 상세 비교
4. **error_analysis**: 에러 유형별 통계
5. **MATH_500_quantization_comparison**: 양자화 방법별 비교 (MATH-500)
6. **AIME_90_quantization_comparison**: 양자화 방법별 비교 (AIME-90)

### LaTeX 논문 사용 예시

```latex
\begin{table}[h]
\centering
\input{reports/tables/overall_performance.tex}
\caption{Overall Performance Comparison of Quantized Models}
\label{tab:overall_performance}
\end{table}
```

---

## 📊 통계 데이터 (statistics/)

### summary.json

전체 통계를 JSON 형식으로 제공:

```json
{
  "total_models": 6,
  "total_datasets": 2,
  "total_responses_analyzed": 3540,
  "models": {
    "모델명": {
      "데이터셋": {
        "accuracy": 0.85,
        "total_samples": 500,
        "correct_count": 425,
        "incorrect_count": 75,
        "avg_word_count": 2983,
        "avg_wait_count": 33.2,
        "token_diversity": 0.25
      }
    }
  }
}
```

### 사용 예시 (Python)

```python
import json

# 통계 로드
with open('reports/statistics/summary.json') as f:
    stats = json.load(f)

# 특정 모델 정확도 추출
baseline_math500 = stats['models']['DeepSeek-R1-Distill-Qwen-1.5B']['MATH-500']['accuracy']
print(f"Baseline MATH-500: {baseline_math500 * 100:.2f}%")
```

---

## 🔬 중간 데이터 (intermediate_data/)

각 모델-데이터셋 조합에 대한 상세 분석 결과:

### JSON 파일
- 전체 분석 결과 (메타데이터 + 응답별 상세 분석)
- 에러 유형, 반복 패턴, 토큰 다양성 등 모든 지표 포함

### CSV 파일
- 응답별 핵심 메트릭만 추출
- Pandas로 쉽게 로드하여 추가 분석 가능

### 파일명 형식
```
{모델명}_{데이터셋}.{json|csv}

예시:
- DeepSeek-R1-Distill-Qwen-1.5B_MATH-500.json
- DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1_AIME-90.csv
```

### CSV 컬럼 구조

| 컬럼 | 설명 |
|------|------|
| idx | 문제 번호 |
| model | 모델 이름 |
| dataset | 데이터셋 이름 |
| is_correct | 정답 여부 (True/False) |
| word_count | 응답 단어 수 |
| wait_count | "Wait," 반복 횟수 |
| token_diversity | 토큰 다양성 (unique/total) |
| token_diversity_last_2k | 마지막 2000 토큰 다양성 |
| has_boxed_answer | `\boxed{}` 형식 답변 존재 여부 |
| gibberish_count | Gibberish 토큰 출현 횟수 |
| repeated_char_count | 반복 문자 패턴 개수 |

### 사용 예시 (Pandas)

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('reports/intermediate_data/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1_MATH-500.csv')

# 오답만 필터링
errors = df[df['is_correct'] == False]

# Severe repetition 케이스
severe_rep = errors[errors['wait_count'] > 1000]
print(f"Severe repetition cases: {len(severe_rep)} / {len(errors)}")

# 평균 통계
print(f"평균 Wait 횟수 (오답): {errors['wait_count'].mean():.1f}")
print(f"평균 토큰 다양성 (오답): {errors['token_diversity'].mean():.4f}")
```

---

## 🎓 논문 작성 가이드

### 추천 구성

1. **Introduction/Related Work**
   - `COMPREHENSIVE_QUANTIZATION_ANALYSIS.md` - Executive Summary 참조

2. **Methods**
   - `tables/overall_performance.*` 인용
   - 양자화 방법 설명 (AWQ, GPTQ, KV-Quant*)

3. **Results**
   - `tables/MATH_500_comparison.*`
   - `tables/AIME_90_comparison.*`
   - `tables/error_analysis.*`

4. **Analysis**
   - `COMPREHENSIVE_QUANTIZATION_ANALYSIS.md` - Section 5, 6 참조
   - Repetition Degeneration 메커니즘 설명

5. **Discussion**
   - `COMPREHENSIVE_QUANTIZATION_ANALYSIS.md` - Section 7 (결론) 참조
   - AWQ vs GPTQ 비교

6. **Conclusion**
   - 양자화 방법 선택 가이드
   - 향후 연구 방향

### 핵심 Figure/Table 추천

**필수 포함**:
1. Table: `overall_performance` (전체 성능 비교)
2. Table: `error_analysis` (에러 패턴 분석)
3. Figure: Repetition Degeneration 다이어그램 (보고서에서 복사)
4. Table: `MATH_500_quantization_comparison` (양자화 방법 비교)

**선택 포함**:
- 문제 유형별 성능 (Section 2.2)
- 응답 길이 분포 (Section 6.3)
- 토큰 다양성 변화 (Section 6.4)

---

## 🔄 데이터 재현

### 분석 재실행

```bash
# 전체 분석
python scripts/analysis/comprehensive_analysis.py \
    --seed 42 \
    --inference_dir ./outputs/inference \
    --output_dir ./analysis_results

# 결과를 reports 폴더로 복사
cp -r analysis_results/* reports/
```

### 개별 모델 분석

```bash
python scripts/analysis/analyze_responses.py \
    --seed 42 \
    --datasets MATH-500 AIME-90 \
    --models DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1
```

---

## 📞 문의 및 추가 분석

추가 분석이 필요한 경우:

1. **통계적 유의성 검증**: `intermediate_data/*.csv` 활용
2. **특정 문제 상세 분석**: `intermediate_data/*.json` → `response_analyses` 배열 확인
3. **커스텀 에러 분석**: CSV 파일 로드 후 Pandas로 분석

---

**분석 날짜**: 2026-01-05
**분석 스크립트**: `scripts/analysis/comprehensive_analysis.py`
**총 분석 샘플**: 3,540개 (6 모델 × 2 데이터셋 × 590개 문제)

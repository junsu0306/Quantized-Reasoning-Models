# SmoothQuant 및 KVQuant 붕괴 현상 분석 보고서

## Collapse Analysis Report for SmoothQuant and KVQuant Models

**분석 대상 모델**: DeepSeek-R1-Distill-Qwen-1.5B
**양자화 방법**: SmoothQuant (W8A8KV8), KVQuant* (KV3, KV4), AWQ, GPTQ
**데이터셋**: MATH-500, AIME-90
**분석 날짜**: 2026-01-07

---

## Executive Summary

### 핵심 발견

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    MATH-500 붕괴율 비교 (낮을수록 안정)                      │
├────────────────────────────────────────────────────────────────────────────┤
│  GPTQ 4-bit    (11.0%)  ████████████                    ✅ 가장 안정        │
│  Baseline      (12.0%)  █████████████                   ✅ 안정            │
│  KVQuant KV4   (12.4%)  █████████████                   ✅ 안정            │
│  GPTQ 3-bit    (13.2%)  ██████████████                  ✅ 안정            │
│  AWQ 4-bit     (13.6%)  ██████████████                  ✅ 안정            │
│  SmoothQuant   (15.0%)  ███████████████                 ⚠️  주의 필요       │
│  KVQuant KV3   (52.0%)  ████████████████████████████████████████████████████ ❌ 불안정 │
│  AWQ 3-bit     (64.2%)  █████████████████████████████████████████████████████████████ ❌ 매우 불안정 │
└────────────────────────────────────────────────────────────────────────────┘
```

### 주요 결론

1. **KVQuant KV3 (52.0% 붕괴율)**: KV Cache를 3-bit로 양자화하면 **심각한 붕괴** 발생
   - AWQ 3-bit과 유사한 수준의 불안정성
   - KV Cache 정밀도가 추론 안정성에 매우 중요함을 시사

2. **KVQuant KV4 (12.4% 붕괴율)**: KV Cache 4-bit는 **Baseline과 거의 동일한 안정성**
   - 메모리 절약하면서 안정성 유지 가능
   - 추천할 만한 양자화 방법

3. **SmoothQuant (15.0% 붕괴율)**: W8A8KV8은 **상대적으로 안정**
   - Baseline(12.0%)보다 약간 높은 붕괴율
   - 프로덕션 사용 가능 수준

4. **AWQ 3-bit (64.2% 붕괴율)**: 가장 불안정한 양자화 방법
   - 답변 생성률 43.0%로 매우 낮음
   - 프로덕션 사용 부적합

---

## 1. 전체 결과 요약

### 1.1 MATH-500 데이터셋 비교

| 모델 | 붕괴율 | 답변 생성률 | 평균 단어 수 | 평균 Wait | 평균 다양성 | 평가 |
|------|--------|-----------|------------|----------|-----------|------|
| **GPTQ 4-bit** | **11.0%** | 96.4% | 2,976 | 36.4 | 0.249 | ✅ 가장 안정 |
| **Baseline (FP16)** | 12.0% | 97.8% | 2,983 | 33.2 | 0.251 | ✅ 안정 |
| **KVQuant KV4** | 12.4% | 97.8% | 3,025 | 39.5 | 0.253 | ✅ 안정 |
| **GPTQ 3-bit** | 13.2% | 94.6% | 2,935 | 41.5 | 0.237 | ✅ 안정 |
| **AWQ 4-bit** | 13.6% | 97.4% | 2,959 | 46.7 | 0.243 | ✅ 안정 |
| **SmoothQuant** | 15.0% | 90.0% | 3,738 | 50.1 | 0.244 | ⚠️ 주의 |
| **KVQuant KV3** | **52.0%** | 88.0% | 5,302 | 144.9 | 0.169 | ❌ 불안정 |
| **AWQ 3-bit** | **64.2%** | 43.0% | 11,327 | 968.7 | 0.119 | ❌ 매우 불안정 |

### 1.2 AIME-90 데이터셋 비교 (고난도)

| 모델 | 붕괴율 | 답변 생성률 | 평균 단어 수 | 평균 Wait | 평균 다양성 |
|------|--------|-----------|------------|----------|-----------|
| **GPTQ 4-bit** | 46.7% | 85.6% | 8,309 | 111.4 | 0.146 |
| **SmoothQuant** | 48.9% | 84.4% | 8,574 | 132.2 | 0.154 |
| **Baseline** | 51.1% | 85.6% | 8,711 | 107.8 | 0.155 |
| **KVQuant KV4** | 57.8% | 81.1% | 9,637 | 123.3 | 0.147 |
| **GPTQ 3-bit** | 57.8% | 73.3% | 9,183 | 97.0 | 0.118 |
| **AWQ 4-bit** | 65.6% | 83.3% | 9,913 | 169.3 | 0.129 |
| **KVQuant KV3** | **91.1%** | 73.3% | 10,596 | 298.2 | 0.114 |
| **AWQ 3-bit** | **93.3%** | 14.4% | 17,312 | 1,531.4 | 0.055 |

> **참고**: AIME-90은 고난도 문제로, 모든 모델에서 붕괴율이 높음. 이는 문제 난이도와 관련됨.

---

## 2. 양자화 방법별 상세 분석

### 2.1 KVQuant* (KV Cache 양자화)

#### KV Cache 비트 수에 따른 안정성 변화

| 지표 | KV4 (4-bit) | KV3 (3-bit) | 변화율 |
|------|------------|------------|-------|
| **MATH-500 붕괴율** | 12.4% | **52.0%** | +319.4% |
| **AIME-90 붕괴율** | 57.8% | **91.1%** | +57.6% |
| **답변 생성률** | 97.8% | 88.0% | -10.0% |
| **평균 Wait 횟수** | 39.5 | 144.9 | +266.8% |
| **평균 토큰 다양성** | 0.253 | 0.169 | -33.2% |

**분석**:
- KV Cache를 3-bit로 양자화하면 **붕괴율이 4배 이상 증가**
- KV4는 Baseline과 거의 동일한 성능 유지
- **KV Cache의 정밀도가 추론 안정성에 핵심적 역할**

#### KVQuant KV3 붕괴 예시 상세

KV3 (3-bit)에서 붕괴가 발생하지만 KV4 (4-bit)에서는 정상 동작하는 케이스 **32개**를 분석했습니다.

---

##### 예시 1: 문장 무한 반복 (확률 문제)

**문제**: A pirate searches seven islands for buried treasure. If each island has a 1/5 chance of having treasure, what is the probability that exactly 4 of the islands have treasure?

| 지표 | KV3 (3-bit) | KV4 (4-bit) |
|------|-------------|-------------|
| **답변** | ❌ 없음 (붕괴) | ✅ 448/15625 |
| **단어 수** | 5,980 | 2,642 |
| **Wait 횟수** | 238 | 51 |
| **다양성** | 0.124 | 0.280 |

**KV3 응답 끝 (붕괴 지점)**:
```
Therefore, 15625=5^6.
Therefore, 15625=5^6.
Therefore, 15625=5^6.
Therefore, 15625=5^6.
Therefore, 15625=5^6.
Therefore, 15625=5^6.
[무한 반복...]
```

**분석**: 이항분포 계산 중 "15625=5^6" 검증 단계에서 같은 문장이 무한 반복. 정답 직전까지 도달했으나 결론 도출 실패.

---

##### 예시 2: 잘못된 추론 결과 (기하 문제)

**문제**: Two runners, A and B, start at a point O on a linear track... An observer at point P... find the maximum of angle APB.

| 지표 | KV3 (3-bit) | KV4 (4-bit) |
|------|-------------|-------------|
| **답변** | ❌ 180° (오답) | ✅ 30° (정답) |
| **단어 수** | 10,602 | 6,293 |
| **Wait 횟수** | 164 | 63 |
| **다양성** | 0.097 | 0.156 |

**KV3 응답 끝**:
```
After detailed analysis, we conclude that the maximum angle APB is achieved
when the runners are positioned such that the angle is 180 degrees, which is
the maximum possible angle between two points in a plane.

\boxed{180^\circ}
```

**분석**: 최적화 문제에서 극값 계산을 잘못 수행. "180°가 최대"라는 **논리적으로 틀린 결론** 도출. KV4는 미분을 통해 정확히 30° 계산.

---

##### 예시 3: 응답 길이 폭발 + 정답 (삼각함수)

**문제**: Simplify (sec x)/(sin x) - (sin x)/(cos x)

| 지표 | KV3 (3-bit) | KV4 (4-bit) |
|------|-------------|-------------|
| **답변** | ✅ cot x | ✅ cot x |
| **단어 수** | 3,727 | 733 |
| **Wait 횟수** | 76 | 2 |
| **다양성** | 0.137 | 0.287 |

**분석**: 정답을 맞추지만 **응답이 5배 이상 길어짐**. 불필요한 재검증과 Wait 반복으로 효율성 크게 저하.

---

##### 예시 4: 완전히 틀린 답 (삼각함수)

**문제**: Given sin∠RPQ = 7/25, what is cos∠RPS?

| 지표 | KV3 (3-bit) | KV4 (4-bit) |
|------|-------------|-------------|
| **답변** | ❌ 16 (오답) | ✅ -24/25 (정답) |
| **단어 수** | 10,825 | 1,555 |
| **Wait 횟수** | 342 | 10 |
| **다양성** | 0.125 | 0.282 |

**KV3 응답 끝**:
```
The sine of the angle is given as 7/25, which corresponds to an angle
of approximately 16°.
Therefore, the measure of angle RPS is 16°.

\boxed{16}
```

**분석**: **문제를 완전히 잘못 이해**. cos 값을 묻는 문제에서 각도를 답으로 제출. 피타고라스 항등식을 사용한 정상적인 풀이 경로를 벗어남.

---

##### 예시 5: 정상적인 풀이 but 긴 응답 (정수론)

**문제**: Positive integers a, b, and 2009, with a<b<2009, form a geometric sequence with an integer ratio. What is a?

| 지표 | KV3 (3-bit) | KV4 (4-bit) |
|------|-------------|-------------|
| **답변** | ✅ 41 | ✅ 41 |
| **단어 수** | 10,361 | 1,467 |
| **Wait 횟수** | 261 | 10 |
| **다양성** | 0.068 | 0.204 |

**분석**: 정답을 맞추지만 **7배 이상 긴 응답**. 소인수분해를 여러 번 반복 검증하며 비효율적 풀이. 실제로 "2009의 소인수분해는 3×13×59"라고 **틀린 정보**(실제: 7²×41)를 중간에 생성했다가 나중에 수정.

---

#### KVQuant KV3 붕괴 특징 요약

| 특징 | 빈도 | 설명 |
|------|------|------|
| **문장 무한 반복** | 높음 | "Therefore, X=Y" 패턴 반복 |
| **응답 길이 폭발** | 매우 높음 | KV4 대비 5-10배 길어짐 |
| **Wait 마커 폭발** | 매우 높음 | 평균 144.9회 (KV4: 39.5회) |
| **잘못된 최종 답** | 중간 | 논리적 오류로 오답 도출 |
| **문제 오해** | 낮음 | 문제 자체를 잘못 이해 |

**결론**: KV Cache 3-bit 양자화는 **Attention 메커니즘의 정밀도 손실**로 인해:
1. 이전 토큰들을 잘못 참조하여 같은 내용 반복
2. 복잡한 수학적 추론에서 논리 오류 발생
3. 정답을 맞추더라도 효율성이 크게 저하

#### KVQuant 붕괴 유형 분포

**KVQuant KV3 (MATH-500)**:
```
no_collapse        48개 (48.0%)  ████████████████████████
wait_loop          28개 (28.0%)  ██████████████
moderate_collapse  21개 (21.0%)  ██████████
severe_collapse     3개 (3.0%)   █
```

**KVQuant KV4 (MATH-500)**:
```
no_collapse        438개 (87.6%)  ████████████████████████████████████████████
wait_loop           29개 (5.8%)   ██
moderate_collapse   25개 (5.0%)   ██
severe_collapse      4개 (0.8%)
```

### 2.2 SmoothQuant (W8A8KV8)

| 지표 | SmoothQuant | Baseline | 비교 |
|------|------------|----------|------|
| **MATH-500 붕괴율** | 15.0% | 12.0% | +25.0% |
| **답변 생성률** | 90.0% | 97.8% | -8.0% |
| **평균 단어 수** | 3,738 | 2,983 | +25.3% |
| **평균 다양성** | 0.244 | 0.251 | -2.8% |

**분석**:
- SmoothQuant는 **Weight, Activation, KV Cache 모두 8-bit**로 양자화
- Baseline 대비 약간의 성능 저하 있으나 **프로덕션 사용 가능 수준**
- 붕괴 유형은 주로 `wait_loop`와 `moderate_collapse`

#### SmoothQuant 붕괴 예시

**Index 0 - severe_collapse (삼각함수)**:
```
문제: Simplify $\tan 100^\circ + 4 \sin 100^\circ.$

응답 끝:
sin theta= (-√3 cos theta)/(1 + cos theta).
Hmm, but that's same as before.
Alternatively, perhaps I can think of this as:
sin theta= (-√3 cos theta)/(1 + cos theta).
Hmm, but that's same as before.  [무한 반복]
```

### 2.3 AWQ 3-bit vs GPTQ 3-bit 비교

| 지표 | AWQ 3-bit | GPTQ 3-bit | 비율 |
|------|----------|-----------|------|
| **MATH-500 붕괴율** | **64.2%** | 13.2% | AWQ 4.9배 |
| **답변 생성률** | 43.0% | 94.6% | GPTQ 2.2배 |
| **평균 Wait 횟수** | 968.7 | 41.5 | AWQ 23.3배 |
| **평균 다양성** | 0.119 | 0.237 | GPTQ 2.0배 |

**핵심 발견**:
- 동일한 3-bit 양자화라도 **GPTQ가 AWQ보다 훨씬 안정적**
- AWQ 3-bit는 프로덕션에 **절대 부적합**
- GPTQ 3-bit는 4-bit과 거의 동일한 안정성

---

## 3. 붕괴 패턴 분석

### 3.1 양자화 방법별 붕괴 유형

| 모델 | severe_collapse | moderate_collapse | wait_loop | mild_collapse |
|------|----------------|-------------------|-----------|---------------|
| **AWQ 3-bit** | 50.4% | 10.8% | 3.0% | - |
| **KVQuant KV3** | 3.0% | 21.0% | 28.0% | - |
| **SmoothQuant** | 6.0% | 3.0% | 5.0% | 1.0% |
| **KVQuant KV4** | 0.8% | 5.0% | 5.8% | 0.8% |
| **GPTQ 3-bit** | 4.6% | 5.0% | 3.6% | - |
| **Baseline** | 1.0% | 5.0% | 5.0% | 1.0% |

### 3.2 문제 유형별 취약성 (MATH-500 기준)

모든 양자화 방법에서 **삼각함수(trigonometry)** 문제가 가장 취약합니다.

| 모델 | 삼각함수 문제 | 삼각함수 붕괴율 |
|------|------------|--------------|
| AWQ 3-bit | 124개 중 | 71.8% |
| KVQuant KV3 | 25개 중 | 68.0% |
| SmoothQuant | 25개 중 | 32.0% |
| KVQuant KV4 | 124개 중 | 14.5% |

### 3.3 붕괴 시 특징적 패턴

#### 패턴 1: 숫자/문자 무한 반복
```
444444444444444444444444444444444444...
z, z, z, z, z, z, z, z, z, z, z, z, z...
```

#### 패턴 2: 문장 무한 반복
```
Therefore, n is co prime to 2,3,4,5.
Therefore, n is co prime to 2,3,4,5.
Therefore, n is co prime to 2,3,4,5.  [무한 반복]
```

#### 패턴 3: Wait 루프
```
Wait, let me verify...
Wait, let me verify...
Wait, let me verify...  [무한 반복]
```

#### 패턴 4: 수학 기호 반복
```
f + f + f + f + f + f + f + f + f + f...
A'B'C' of A'B'C' of A'B'C' of A'B'C'...
```

---

## 4. 결론 및 권장사항

### 4.1 양자화 방법 선택 가이드

| 시나리오 | 추천 방법 | 붕괴율 | 메모리 절약 | 비고 |
|---------|---------|--------|-----------|------|
| **정확도 최우선** | GPTQ 4-bit | 11.0% | 중간 | 가장 안정적 |
| **KV Cache 메모리 절약** | KVQuant KV4 | 12.4% | 높음 | Baseline 수준 안정성 |
| **전체 메모리 절약** | SmoothQuant | 15.0% | 높음 | 프로덕션 사용 가능 |
| **극한 메모리 절약** | GPTQ 3-bit | 13.2% | 매우 높음 | AWQ보다 훨씬 안정 |
| **비권장** | KVQuant KV3 | 52.0% | - | 심각한 불안정 |
| **절대 비권장** | AWQ 3-bit | 64.2% | - | 프로덕션 부적합 |

### 4.2 핵심 발견 요약

1. **KV Cache 비트 수가 추론 안정성에 핵심**
   - KV4 (12.4%) vs KV3 (52.0%): 3-bit로 내리면 붕괴율 4배 증가
   - Weight 양자화보다 KV Cache 양자화가 더 민감

2. **GPTQ가 AWQ보다 3-bit에서 훨씬 안정적**
   - 동일 3-bit: GPTQ (13.2%) vs AWQ (64.2%)
   - GPTQ는 4-bit과 거의 동일한 성능 유지

3. **SmoothQuant (W8A8KV8)은 안정적**
   - 8-bit 양자화는 추론에 충분한 정밀도 제공
   - Baseline 대비 약 3%p 높은 붕괴율

4. **고난도 문제에서 모든 모델이 취약**
   - AIME-90에서 Baseline도 51.1% 붕괴
   - 문제 난이도 자체가 붕괴의 원인이 될 수 있음

### 4.3 향후 연구 방향

1. **KV Cache 양자화 개선**
   - KV3의 불안정성 원인 분석
   - Mixed-precision KV Cache (예: 일부 레이어만 4-bit)

2. **AWQ 알고리즘 개선**
   - GPTQ 대비 3-bit에서 불안정한 원인 분석
   - AWQ의 활성화 인식 양자화가 3-bit에서 한계

3. **붕괴 조기 탐지 및 중단**
   - Wait 횟수, 토큰 다양성 모니터링
   - 붕괴 징후 감지 시 응답 조기 종료

---

## 부록: 데이터 및 재현성

### A.1 분석 스크립트

```bash
python scripts/analysis/smoothquant_kvquant_collapse_analysis.py \
    --inference_dir ./outputs/inference \
    --output_dir ./reports
```

### A.2 생성된 파일

```
reports/
├── SMOOTHQUANT_KVQUANT_COLLAPSE_REPORT.md  # 본 보고서
└── smoothquant_kvquant_analysis.json       # 상세 분석 데이터
```

### A.3 데이터 출처

| 모델 | 디렉토리 |
|------|---------|
| Baseline | DeepSeek-R1-Distill-Qwen-1.5B-seed42 |
| SmoothQuant | DeepSeek-R1-Distill-Qwen-1.5B-smoothquant-w8a8kv8-tp1-seed42 |
| KVQuant KV4 | DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1-seed42 |
| KVQuant KV3 | DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv3-tp1-seed42 |
| AWQ 3-bit | DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1-seed42 |
| AWQ 4-bit | DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1-seed42 |
| GPTQ 3-bit | DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1-seed42 |
| GPTQ 4-bit | DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1-seed42 |

---

**보고서 생성**: 2026-01-07
**분석 도구**: scripts/analysis/smoothquant_kvquant_collapse_analysis.py

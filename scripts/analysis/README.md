# Response Analysis Scripts

양자화된 모델들의 수학 문제 풀이 응답을 분석하는 스크립트입니다.

## 파일 구성

- `run_all_quantized_models.sh`: 모든 양자화 모델에 대해 추론을 실행하는 배치 스크립트
- `analyze_responses.py`: 응답의 오류와 이상을 감지하고 분석하는 스크립트

## 사용법

### 1. 모든 양자화 모델에 대해 추론 실행

```bash
# 기본 실행 (device=0, seed=42)
bash scripts/analysis/run_all_quantized_models.sh

# 특정 디바이스와 시드 지정
bash scripts/analysis/run_all_quantized_models.sh 0 42
```

이 스크립트는 다음 모델들에 대해 MATH-500과 AIME-90 데이터셋으로 추론을 실행합니다:
- 원본 FP16 모델
- AWQ (w3, w4)
- GPTQ (w3, w4)
- KVQuant* (kv3, kv4)
- SmoothQuant (w8a8kv8)
- FlatQuant (w4a4kv4)

### 2. 응답 분석

추론이 완료된 후, 응답을 분석합니다:

```bash
# 기본 분석
python scripts/analysis/analyze_responses.py --seed 42

# 상세 리포트 저장
python scripts/analysis/analyze_responses.py --seed 42 --save_report

# 특정 데이터셋만 분석
python scripts/analysis/analyze_responses.py --datasets MATH-500 --seed 42

# 더 많은 예시 출력
python scripts/analysis/analyze_responses.py --seed 42 --max_examples 5
```

## 분석 항목

응답 분석 스크립트는 다음 항목들을 자동으로 감지합니다:

### 1. **불완전한 응답 (Incomplete Responses)**
- 답변 태그 누락 (`\boxed{}`, `####` 등)
- 너무 짧은 응답 (< 50자)
- 비정상적인 종료 (구두점 없이 끝남)

### 2. **깨진/왜곡된 텍스트 (Garbled Text)**
- 반복되는 토큰 (같은 단어가 5회 이상 반복)
- 반복되는 문자 (같은 문자가 10회 이상 반복)
- 비정상적인 특수문자 비율 (30% 이상)
- 인코딩 오류 (mojibake, �, \ufffd 등)

### 3. **추론 오류 (Reasoning Errors)**
- 수학적 모순 (예: 5 = 3)
- 의미 없는 텍스트 (인식 가능한 단어가 너무 적음)

### 4. **길이 이상 (Length Anomalies)**
- 평균 응답 길이에서 3 표준편차 이상 벗어남

## 출력 예시

### 요약 테이블
```
====================================================================================================
ERROR DETECTION SUMMARY
====================================================================================================
Model                                                        Dataset         Total    Incomplete   Garbled    Reasoning    Length
----------------------------------------------------------------------------------------------------
DeepSeek-R1-Distill-Qwen-1.5B                                MATH-500        500      2            0          1            3
DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1                 MATH-500        500      5            2          3            5
...
```

### 상세 오류 예시
각 모델에 대해 최대 N개 (기본값: 3)의 상세한 오류 예시를 출력합니다:
- 오류 유형
- 정답 (gold answer)
- 평가 지표
- 생성된 텍스트 미리보기 (처음 500자)

### JSON 리포트
`--save_report` 옵션을 사용하면 `./outputs/inference/error_report_seed{seed}.json`에 상세한 리포트가 저장됩니다.

## 커스터마이징

### 다른 모델 추가

`run_all_quantized_models.sh`의 `quantized_models` 배열에 모델 경로를 추가하세요:

```bash
quantized_models=(
    ...
    "./outputs/modelzoo/your_method/DeepSeek-R1-Distill-Qwen-1.5B-your_method-tp1"
)
```

### 다른 데이터셋 추가

`run_all_quantized_models.sh`의 `datasets` 배열에 데이터셋을 추가하세요:

```bash
datasets=("MATH-500" "GSM8K" "AIME-90" "GPQA-Diamond")
```

### 분석 규칙 수정

`analyze_responses.py`의 감지 함수들을 수정하여 커스터마이징할 수 있습니다:
- `detect_incomplete_response()`: 불완전한 응답 감지 규칙
- `detect_garbled_text()`: 깨진 텍스트 감지 규칙
- `detect_reasoning_errors()`: 추론 오류 감지 규칙
- `detect_length_anomalies()`: 길이 이상 감지 규칙

## 예상 실행 시간

- DeepSeek-R1-Distill-Qwen-1.5B 모델 기준
- MATH-500 + AIME-90 (약 590 샘플)
- 모델당 약 15분 ~ 30분
- 전체 9개 모델: 약 2.5~4.5시간

## 주의사항

1. **GPU 메모리**: 1.5B 모델은 TP=1로 실행되므로 단일 GPU에서 실행 가능합니다.
2. **디스크 공간**: 결과 파일이 상당히 클 수 있습니다 (~수백 MB).
3. **기존 결과 보존**: 기본적으로 기존 결과가 있으면 건너뜁니다. 재실행하려면 `--overwrite` 플래그를 inference.py에 추가하세요.

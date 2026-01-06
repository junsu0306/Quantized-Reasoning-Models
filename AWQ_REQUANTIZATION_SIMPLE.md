# AWQ 재양자화 (수학 데이터 사용) - 간단 가이드

## 🔴 문제 요약

| 방법 | 기존 캘리브레이션 | 시퀀스 길이 | 파일 |
|------|---------------|-----------|------|
| **AWQ** | ❌ Pile (일반 텍스트) | 512 | `outputs/modelzoo/awq/*-tp1` |
| **GPTQ** | ✅ NuminaMath (수학) | 2048 | `outputs/modelzoo/gptq/*-tp1` |

→ **불공정한 비교!**

## ✅ 해결: 기존 스크립트 활용

**새 스크립트**: `scripts/quantization/awq_with_math_data.sh`
- 기존 잘 작동하는 AWQ 코드 활용
- 캘리브레이션 데이터만 변경: `pileval` → `reasoning-numina-math-1.5`
- 시퀀스 길이 변경: `512` → `2048`

## 🚀 실행 방법

```bash
# AWQ 3-bit, 4-bit 재양자화 (수학 데이터)
bash scripts/quantization/awq_with_math_data.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B \
    1 \
    0

# 파라미터:
# 1. 모델 경로
# 2. Tensor Parallel (TP) - 1로 설정
# 3. GPU 디바이스 번호 - 0
```

실행 시간: 약 30-60분 (3-bit, 4-bit 순차 실행)

## 📂 생성되는 모델

```
outputs/modelzoo/awq/
├── DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-mathcalib-tp1/  # ← 새로 생성
├── DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-mathcalib-tp1/  # ← 새로 생성
├── DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1/  # (기존, Pile)
└── DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1/  # (기존, Pile)
```

`-mathcalib-` 접미사로 구분됩니다.

## 📊 재실험

재양자화 후 MATH-500으로 평가:

```bash
# AWQ 3-bit (수학 캘리브레이션)
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-mathcalib-tp1 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-mathcalib-seed42

# AWQ 4-bit (수학 캘리브레이션)
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-mathcalib-tp1 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-mathcalib-seed42
```

## 📈 예상 결과

수학 데이터 캘리브레이션 후:

| 지표 | 기존 (Pile) | 예상 (NuminaMath) |
|------|-----------|------------------|
| **AWQ 3-bit 정확도** | 52.6% | **?% (향상 예상)** |
| **Wait 마커** | 1,278회 | **?회 (감소 예상)** |
| **응답 길이** | 16,314 단어 | **?단어 (감소 예상)** |
| **AWQ vs GPTQ 격차** | 2.1배 | **?배 (감소 예상)** |

## 🔍 비교 분석

재실험 후 다시 비교:

```bash
# 비교 스크립트 재실행
python scripts/analysis/compare_3bit_vs_4bit.py
```

## ⚠️ 주의사항

1. **디스크 공간**: 모델당 약 3-4GB 필요
2. **GPU 메모리**: 양자화 시 약 16GB+ 필요
3. **기존 모델 보존**: `-mathcalib-` 접미사로 구분되어 기존 모델 유지됨
4. **실행 시간**: 전체 약 30-60분 소요

## 🆚 기존 vs 새 방법 비교

| 항목 | 기존 스크립트 | 새 스크립트 |
|------|------------|-----------|
| **스크립트** | `scripts/quantization/awq.sh` | `scripts/quantization/awq_with_math_data.sh` |
| **캘리브레이션** | Pile | NuminaMath |
| **시퀀스 길이** | 512 | 2048 |
| **출력 경로** | `*-tp1` | `*-mathcalib-tp1` |
| **실행 코드** | 동일 (`methods/awq/run_awq.py`) | 동일 |

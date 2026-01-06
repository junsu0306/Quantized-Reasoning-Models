# AWQ/GPTQ 재양자화 가이드 - 수학 데이터 사용

## 🔴 문제점 발견

기존 실험에서 **AWQ와 GPTQ가 서로 다른 캘리브레이션 데이터를 사용**하여 공정한 비교가 아니었습니다:

| 방법 | 기존 캘리브레이션 데이터 | 문제점 |
|------|----------------------|--------|
| **AWQ** | Pile (일반 텍스트) | 수학 문제와 무관한 데이터 |
| **GPTQ** | NuminaMath-1.5 (수학 데이터) | 수학 문제에 특화된 데이터 |

이는 **불공정한 비교**입니다!

## ✅ 해결 방법

모든 양자화 방법에 **동일한 수학 데이터(NuminaMath-1.5)**를 사용하도록 코드 수정:

### 변경된 파일

1. **`real_quantization/calib_data.py`**
   - `get_reasoning_calib_text_list()` 함수 추가
   - AWQ용 텍스트 리스트 형식 캘리브레이션 데이터 제공

2. **`real_quantization/real_quantization.py`**
   - AWQ-autoawq: Pile → NuminaMath로 변경
   - AWQ-llmcompressor: Pile → NuminaMath로 변경
   - 3-bit 양자화 지원 추가 (`choices=[3, 4]`)
   - 시퀀스 길이 통일: 512 → 2048 (GPTQ와 동일)

3. **`scripts/real_quantization/requantize_with_math_data.sh`**
   - 새로 생성: 4개 모델(AWQ 3/4bit, GPTQ 3/4bit) 일괄 재양자화

## 🚀 재양자화 실행

### 1. 사전 준비

캘리브레이션 데이터가 있는지 확인:
```bash
ls ./datasets/gen_data/DeepSeek-R1-Distill-Qwen-1.5B/NuminaMath-1.5.jsonl
```

없으면 먼저 생성:
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset NuminaMath-1.5 \
    --max_samples 256 \
    --output_dir ./datasets/gen_data/DeepSeek-R1-Distill-Qwen-1.5B
```

### 2. 재양자화 실행

```bash
# GPU 0 사용
bash scripts/real_quantization/requantize_with_math_data.sh 0
```

실행 순서:
1. AWQ 4-bit (수학 데이터)
2. AWQ 3-bit (수학 데이터)
3. GPTQ 4-bit (수학 데이터)
4. GPTQ 3-bit (수학 데이터)

### 3. 생성된 모델 확인

```bash
ls -lh ./outputs/modelzoo/real_quantization/
```

예상 출력:
```
awq-autoawq/DeepSeek-R1-Distill-Qwen-1.5B-quantized.awq-autoawq-w4g128/
awq-autoawq/DeepSeek-R1-Distill-Qwen-1.5B-quantized.awq-autoawq-w3g128/
gptq-gptqmodel/DeepSeek-R1-Distill-Qwen-1.5B-quantized.gptq-gptqmodel-w4g128/
gptq-gptqmodel/DeepSeek-R1-Distill-Qwen-1.5B-quantized.gptq-gptqmodel-w3g128/
```

## 📊 재실험 필요

재양자화 후 다시 추론 실행:

```bash
# AWQ 4-bit
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/real_quantization/awq-autoawq/DeepSeek-R1-Distill-Qwen-1.5B-quantized.awq-autoawq-w4g128 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-mathcalib-seed42

# AWQ 3-bit
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/real_quantization/awq-autoawq/DeepSeek-R1-Distill-Qwen-1.5B-quantized.awq-autoawq-w3g128 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-mathcalib-seed42

# GPTQ 4-bit
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/real_quantization/gptq-gptqmodel/DeepSeek-R1-Distill-Qwen-1.5B-quantized.gptq-gptqmodel-w4g128 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-mathcalib-seed42

# GPTQ 3-bit
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model ./outputs/modelzoo/real_quantization/gptq-gptqmodel/DeepSeek-R1-Distill-Qwen-1.5B-quantized.gptq-gptqmodel-w3g128 \
    --dataset MATH-500 \
    --output_dir ./outputs/inference/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-mathcalib-seed42
```

## 🔍 예상 결과

수학 데이터로 캘리브레이션한 후:
- **AWQ 3-bit 정확도 향상 예상**: 52.6% → ??%
- **Repetition degeneration 감소 예상**: Wait 1,278회 → ??회
- **AWQ vs GPTQ 격차 감소 예상**: 현재 2.1배 차이 → ??배

## 📝 기술적 세부사항

### 캘리브레이션 설정 통일

| 파라미터 | 기존 AWQ | 기존 GPTQ | 수정 후 (공통) |
|---------|---------|-----------|---------------|
| **데이터셋** | Pile | NuminaMath | **NuminaMath** |
| **샘플 수** | 128 | 128 | **128** |
| **시퀀스 길이** | 512 | 2048 | **2048** |
| **그룹 크기** | 128 | 128 | **128** |
| **비대칭 양자화** | True | True | **True** |

### 코드 변경 요약

```python
# 이전 (AWQ)
model.quantize(
    calib_data="./datasets/pile-val-backup",  # ❌ 일반 텍스트
    max_calib_seq_len=512,  # ❌ 짧은 시퀀스
)

# 수정 후 (AWQ)
calib_text_list = get_reasoning_calib_text_list(
    model_name=args.model_name,
    n_samples=128
)
model.quantize(
    calib_data=calib_text_list,  # ✅ 수학 데이터
    max_calib_seq_len=2048,  # ✅ GPTQ와 동일
)
```

## ⚠️ 주의사항

1. **NuminaMath 데이터 필수**: 재양자화 전에 `datasets/gen_data/` 폴더에 NuminaMath-1.5.jsonl 파일이 있어야 합니다.
2. **GPU 메모리**: 양자화는 모델 로딩이 필요하므로 충분한 GPU 메모리 필요 (16GB+ 권장)
3. **시간**: 모델당 약 10-30분 소요 (총 1-2시간)
4. **기존 모델 백업**: 재양자화 전에 기존 모델을 백업하는 것을 권장

## 📚 참고

- AWQ 논문: https://arxiv.org/abs/2306.00978
- GPTQ 논문: https://arxiv.org/abs/2210.17323
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- GPTQModel: https://github.com/ModelCloud/GPTQModel

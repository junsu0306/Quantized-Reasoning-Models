#!/bin/bash

# AWQ와 GPTQ를 동일한 수학 데이터로 재양자화
# 공정한 비교를 위해 AWQ도 수학 데이터(NuminaMath)로 캘리브레이션

MODEL="./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE=${1:-0}  # 기본값 GPU 0

echo "=================================================="
echo "수학 데이터로 AWQ/GPTQ 재양자화"
echo "=================================================="
echo "Model: ${MODEL}"
echo "Device: ${DEVICE}"
echo ""

# AWQ 4-bit
echo "🔥 [1/4] AWQ 4-bit 양자화 (수학 데이터)"
CUDA_VISIBLE_DEVICES=${DEVICE} python -m real_quantization.real_quantization \
    --model ${MODEL} \
    --method awq-autoawq \
    --w_bits 4 --w_groupsize 128 --w_asym

echo ""

# AWQ 3-bit
echo "🔥 [2/4] AWQ 3-bit 양자화 (수학 데이터)"
CUDA_VISIBLE_DEVICES=${DEVICE} python -m real_quantization.real_quantization \
    --model ${MODEL} \
    --method awq-autoawq \
    --w_bits 3 --w_groupsize 128 --w_asym

echo ""

# GPTQ 4-bit
echo "🔥 [3/4] GPTQ 4-bit 양자화 (수학 데이터)"
CUDA_VISIBLE_DEVICES=${DEVICE} python -m real_quantization.real_quantization \
    --model ${MODEL} \
    --method gptq-gptqmodel \
    --w_bits 4 --w_groupsize 128 --w_asym

echo ""

# GPTQ 3-bit
echo "🔥 [4/4] GPTQ 3-bit 양자화 (수학 데이터)"
CUDA_VISIBLE_DEVICES=${DEVICE} python -m real_quantization.real_quantization \
    --model ${MODEL} \
    --method gptq-gptqmodel \
    --w_bits 3 --w_groupsize 128 --w_asym

echo ""
echo "=================================================="
echo "✅ 모든 양자화 완료!"
echo "=================================================="
echo ""
echo "생성된 모델:"
ls -lh ./outputs/modelzoo/real_quantization/*/DeepSeek-R1-Distill-Qwen-1.5B-quantized.*

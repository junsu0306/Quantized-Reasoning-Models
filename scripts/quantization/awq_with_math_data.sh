#!/bin/bash

# AWQ 양자화 - 수학 데이터(NuminaMath) 사용
# GPTQ와 동일한 캘리브레이션 데이터 사용

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B
tp=${2}     # 1
device=${3} # 0

model_name=$(basename "$model")

echo "=================================================="
echo "AWQ 양자화 (수학 데이터 캘리브레이션)"
echo "=================================================="
echo "Model: ${model}"
echo "TP: ${tp}"
echo "Device: ${device}"
echo "Calibration: reasoning-numina-math-1.5 (GPTQ와 동일)"
echo "Sequence Length: 2048 (GPTQ와 동일)"
echo ""

bits=("3" "4")
for BITS in "${bits[@]}"; do
    echo "🔥 양자화: ${BITS}-bit"
    CUDA_VISIBLE_DEVICES=${device} \
    python -m methods.awq.run_awq \
        --model ${model} \
        --w_bits ${BITS} --w_groupsize 128 --w_asym \
        --calib_data reasoning-numina-math-1.5 \
        --seqlen 2048 \
        --save_qmodel_path ./outputs/modelzoo/awq/${model_name}-awq-w${BITS}g128-mathcalib-tp${tp}
    echo ""
done

echo "=================================================="
echo "✅ 완료!"
echo "=================================================="
echo "생성된 모델:"
ls -lh ./outputs/modelzoo/awq/${model_name}-awq-w*mathcalib*

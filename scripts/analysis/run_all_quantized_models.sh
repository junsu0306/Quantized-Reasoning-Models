#!/bin/bash

# Run inference on all quantized models for DeepSeek-R1-Distill-Qwen-1.5B
# Usage: bash scripts/analysis/run_all_quantized_models.sh [device] [seed]

# Activate conda environment
source ~/anaconda3/bin/activate quantized-reasoning-models

device=${1:-0}
seed=${2:-42}

# Datasets to evaluate
datasets=("MATH-500" "AIME-90")

# Base model path
base_model="./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B"

# List of quantized models (all with tp1 for 1.5B model)
quantized_models=(
    "./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1"
    "./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1"
    "./outputs/modelzoo/gptq/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1"
    "./outputs/modelzoo/gptq/DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1"
    "./outputs/modelzoo/kvquant_star/DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1"
    "./outputs/modelzoo/kvquant_star/DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv3-tp1"
    "./outputs/modelzoo/smoothquant/DeepSeek-R1-Distill-Qwen-1.5B-smoothquant-w8a8kv8-tp1"
    "./outputs/modelzoo/flatquant/DeepSeek-R1-Distill-Qwen-1.5B-flatquant-w4a4kv4-tp1"
)

# Run inference on base model (FP16)
echo "===== Running inference on base model (FP16) ====="
for dataset in "${datasets[@]}"; do
    echo "Dataset: $dataset"

    # Set max_samples based on dataset
    if [[ "$dataset" == "MATH-500" ]]; then
        max_samples=100
    elif [[ "$dataset" == "AIME-90" ]]; then
        max_samples=45
    fi

    CUDA_VISIBLE_DEVICES=${device} \
    python -m inference \
        --model ${base_model} \
        --dataset ${dataset} \
        --seed ${seed} \
        --max_samples ${max_samples}
done

# Run inference on all quantized models
for model_path in "${quantized_models[@]}"; do
    echo "===== Running inference on $(basename ${model_path}) ====="

    # Check if model exists
    if [ ! -d "${model_path}" ]; then
        echo "Model not found: ${model_path}, skipping..."
        continue
    fi

    for dataset in "${datasets[@]}"; do
        echo "Dataset: $dataset"

        # Set max_samples based on dataset
        if [[ "$dataset" == "MATH-500" ]]; then
            max_samples=100
        elif [[ "$dataset" == "AIME-90" ]]; then
            max_samples=45
        fi

        CUDA_VISIBLE_DEVICES=${device} \
        python -m inference \
            --model ${model_path} \
            --dataset ${dataset} \
            --seed ${seed} \
            --max_samples ${max_samples}
    done
done

echo "===== All inference runs completed ====="
echo "Results saved in ./outputs/inference"
echo ""
echo "To analyze the responses, run:"
echo "python scripts/analysis/analyze_responses.py --seed ${seed}"

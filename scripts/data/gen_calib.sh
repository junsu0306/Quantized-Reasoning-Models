#!/bin/bash

model=$1    # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
devices=$2  # 0,1,2,3

model_name=$(basename "$model")

# Set LD_LIBRARY_PATH for torch
export LD_LIBRARY_PATH=/home/junsu/anaconda3/envs/quantized-reasoning-models/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=${devices} \
python inference.py \
    --model $model \
    --dataset NuminaMath-1.5 \
    --max_samples 256 \
    --output_dir ./datasets/gen_data/${model_name}

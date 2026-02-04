#!/bin/bash
# Pangu model pruning script - Supports NPU execution
# Usage: bash scripts/pangu_v1.sh

# Set Huggingface mirror (optional, uncomment as needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_MIRROR=https://hf-mirror.com

# Set NPU device
export ASCEND_RT_VISIBLE_DEVICES=0

python main.py \
    --model ./cache/llm_weights/openPangu-Embedded-7B \
    --sparsity_ratio $sparsity \
    --prune_method d2prune \
    --sparsity_type unstructured \
    --cali_dataset c4 \
    --cali_data_path ../cache/data/c4 \
    --eval_dataset wikitext2 \
    --eval_data_path ../cache/data/wikitext \
    --output_dir out/pangu-admm-beta0.9-sp${sparsity}/ \
    --s 1500 \
    --r1 1 \
    --r2 0 \
    --beta 0.9 \
    --d2_wanda \
    --d2_admm \
    --target_layer_names "['self_attn.q_proj']" \
    --device npu:$ASCEND_RT_VISIBLE_DEVICES \
    --eval_zero_shot \
    --free
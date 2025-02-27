#!/bin/bash

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=model
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks eval.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
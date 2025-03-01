#!/bin/bash

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# MODEL="$LOCAL_DIR/temp/Qwen2.5-3B-GRPO"
MODEL="$LOCAL_DIR/models/Qwen2.5-3B"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,use_chat_template=true"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks fine-tune/deepseek_r1/eval.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
    
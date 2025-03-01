#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL="$LOCAL_DIR/temp/Qwen2.5-3B-GRPO"
# MODEL="$LOCAL_DIR/models/Qwen2.5-3B"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=4,use_chat_template=true"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks fine-tune/deepseek_r1/eval.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR


lighteval vllm \
    "$SCRIPT_DIR/vllm_model_config.yaml" \
    "custom|$TASK|0|0" \
    --custom-tasks fine-tune/deepseek_r1/eval.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt="Respond in the following format: \n\n<question>\n...\n</question>\n\n<reasoning>\n...\n</reasoning>\n\n<answer>\n...\n</answer>"
#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/env.sh"

if [ -z "$LOCAL_DIR" ]; then
    echo "LOCAL_DIR is not defined"
    exit 1
fi

if [ ! -d "$LOCAL_DIR/models/Qwen2.5-3B" ]; then
    hfd.sh unsloth/Qwen2.5-3B --local-dir $LOCAL_DIR/models/Qwen2.5-3B
fi

if [ ! -d "$LOCAL_DIR/models/Qwen2.5-3B-Instruct" ]; then
    hfd.sh unsloth/Qwen2.5-3B-Instruct --local-dir $LOCAL_DIR/models/Qwen2.5-3B-Instruct
fi

if [ ! -d "$LOCAL_DIR/datasets/gsm8k" ]; then
    hfd.sh openai/gsm8k --dataset --local-dir $LOCAL_DIR/datasets/gsm8k
fi


if [ ! -d "$LOCAL_DIR/datasets/aime_2024" ]; then
    hfd.sh HuggingFaceH4/aime_2024 --dataset --local-dir $LOCAL_DIR/datasets/aime_2024
fi

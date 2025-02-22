#!/usr/bin/env bash


if [ ! -d "$LOCAL_DIR" ]; then
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x ./hfd.sh
    move ./hfd.sh $LOCAL_BIN
    pip install aria2
fi


hfd.sh unsloth/Qwen2.5-3B --local-dir $LOCAL_DIR/models

hfd.sh openai/gsm8k --dataset --local-dir $LOCAL_DIR/datasets


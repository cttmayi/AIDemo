#!/bin/bash
script_dir=$(dirname "$(realpath "$0")")
file_name="src/do_training.py"
full_path="${script_dir}/${file_name}"

# 打印连接后的完整路径
echo "Run python script: $full_path"

python "$full_path" \
    --model_name_or_path "model/opt-125m" \
    --dataset_name_or_path "data/text" \
    --output_dir "model/opt-125m_pretrain" \
    --use_peft_lora True \
    --num_train_epochs 5.0



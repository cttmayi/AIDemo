#!/bin/bash
script_dir=$(dirname "$(realpath "$0")")
file_name="src/do_test.py"
full_path="${script_dir}/${file_name}"

# 打印连接后的完整路径
echo "Run python script: $full_path"

python "$full_path" \
    --model_name_or_path model/opt-350m_sft \
    --dataset_name_or_path data/text \
    --template_format example

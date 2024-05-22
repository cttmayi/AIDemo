#!/bin/bash
script_dir=$(dirname "$(realpath "$0")")
file_name="src/do_model.py"
full_path="${script_dir}/${file_name}"

# 打印连接后的完整路径
echo "Run python script: $full_path"

python "$full_path" \
    --model_name "facebook/opt-125m" \
    --model_path "model/opt-125m"
#!/bin/bash
script_dir=$(dirname "$(realpath "$0")")
file_name="src/do_dataset.py"
full_path="${script_dir}/${file_name}"

# 打印连接后的完整路径
echo "Run python script: $full_path"

python "$full_path" \
    --data_path "${script_dir}/data/text.jsonl" \
    --save_path "data/text" \
    --save_format "jsonl"

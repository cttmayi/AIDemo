#!/bin/bash

# script
script_dir=$(dirname "$(realpath "$0")")
do_dataset="${script_dir}/src/do_dataset.py"
do_training="${script_dir}/src/do_training.py"


model="Qwen/Qwen1.5-1.8B"
dataset_local_path="data/45k_python_code_chinese_instruction"

# download dataset
if [ ! -d ${dataset_local_path} ]; then
    python "$do_dataset" \
        --data_path jean1/45k_python_code_chinese_instruction \
        --save_path ${dataset_local_path} \
        --test_size 0.05
fi

# fine-tune
python "$do_training" \
    --model_name_or_path ${model} \
    --dataset_name_or_path ${dataset_local_path} \
    --template_format coding \
    --max_seq_length 128 \
    --output_dir "model/qwen_sft_peft" \
    --use_peft_lora True \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --num_train_epochs 1.0
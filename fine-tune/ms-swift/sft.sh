# sh examples/custom/sft.sh

current_dir=$(cd "$(dirname "$0")" || exit; pwd)

register_path="${current_dir}/register.py"
model_name="Qwen/Qwen3-0.6B-Base"


swift sft \
    --custom_register_path "${register_path}" \
    --model "${model_name}" \
    --train_type lora \
    --dataset stsb \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 2048 \
    --output_dir output \
    --dataset_num_proc 4

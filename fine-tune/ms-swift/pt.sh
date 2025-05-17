# If not using flash_attn, or transformers<4.44,
# or encountering an abnormally large loss (i.e., the model does not support packing),
# please remove `--packing true`.
current_dir=$(cd "$(dirname "$0")" || exit; pwd)

register_path="${current_dir}/register.py"
model_name="Qwen/Qwen3-0.6B-Base"


swift pt \
    --custom_register_path "${register_path}" \
    --model "${model_name}" \
    --train_type lora \
    --dataset pt_text \
    --torch_dtype float32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 256 \
    --max_steps 10 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true \
    --output_dir "output/${model_name}" \


#    --packing true \
#    --streaming true \

#    --deepspeed zero3 \
#    --attn_impl flash_attn
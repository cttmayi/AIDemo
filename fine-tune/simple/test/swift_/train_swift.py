# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)

model_type = ModelType.qwen2_0_5b

sft_args = SftArguments(
    model_type=model_type,
    dataset=[f'{DatasetName.blossom_math_zh}#2000'],
    output_dir='output')

result = sft_main(sft_args)

best_model_checkpoint = result['best_model_checkpoint']

print(f'best_model_checkpoint: {best_model_checkpoint}')
# torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True)
# merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)
# torch.cuda.empty_cache()

app_ui_main(infer_args)
from unsloth import FastLanguageModel
import os
from transformers.trainer_utils import get_last_checkpoint

local_dir = os.environ['LOCAL_DIR'] # '/hy-tmp'

local_model_name = local_dir + "/models/Qwen2.5-3B" # "Qwen/Qwen2.5-3B-Instruct",
local_data_name = local_dir + "/datasets/gsm8k" # "openai/gsm8k"
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
output_dir = local_dir + "/outputs/Qwen2.5-3B-GRPO"
save_model_dir = local_dir + "/temp/Qwen2.5-3B-GRPO"

resume_from_checkpoint = get_last_checkpoint(output_dir)
# local_model_name = resume_from_checkpoint if resume_from_checkpoint is not None else local_model_name

# Load and prep model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
if tokenizer.chat_template is None:
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

model.load_adapter(
    resume_from_checkpoint, 
    is_trainable=False)


# Merge to 16bit
if True: model.save_pretrained_merged(save_model_dir, tokenizer, save_method = "merged_16bit",)

# Merge to 4bit
if False: model.save_pretrained_merged(save_model_dir, tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
if False: model.save_pretrained_merged(save_model_dir, tokenizer, save_method = "lora",)

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf(save_model_dir, tokenizer,)

# Save to 16bit GGUF
if False: model.save_pretrained_gguf(save_model_dir, tokenizer, quantization_method = "f16")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf(save_model_dir, tokenizer, quantization_method = "q4_k_m")
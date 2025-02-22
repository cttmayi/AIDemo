import os
local_dir = os.environ['LOCAL_DIR'] # '/hy-tmp'

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
import re, os
from datasets import load_dataset, Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer


local_model_name = local_dir + "/models/Qwen2.5-3B" # "Qwen/Qwen2.5-3B-Instruct",
local_data_name = local_dir + "/datasets/gsm8k" # "openai/gsm8k"
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
output_dir = local_dir + "/outputs/Qwen2.5-3B-GRPO"
# num_train_epochs = 0.01
max_steps = 50
save_steps = 50
save_total_limit = 1

##########################################

def get_checkpoint_number(path):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    if path is not None:
        path = os.path.basename(path)
        if _re_checkpoint.search(path) is not None:
            return int(_re_checkpoint.search(path).groups()[0])
    return 0


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset(local_data_name, 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

if __name__ == "__main__":
    checkpoint = None
    if os.path.exists(output_dir):
        checkpoint = get_last_checkpoint(output_dir)
    print(f"Checkpoint: {checkpoint}")

    max_steps = get_checkpoint_number(checkpoint) + max_steps

    # Load and prep model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = local_model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    dataset = get_gsm8k_questions()

    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        # num_train_epochs = num_train_epochs,
        max_steps = max_steps,
        save_steps = save_steps,
        save_total_limit= save_total_limit, # Keep only the last N checkpoints
        max_grad_norm = 0.1,
        report_to = "tensorboard", # Can use Weights & Biases
        output_dir = output_dir,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    # model.save_lora("grpo_saved_lora")

    if True:
        from vllm import SamplingParams
        question = "How many r's are in strawberry?"
        text = tokenizer.apply_chat_template([
            {"role" : "user", "content" : question},
        ], tokenize = False, add_generation_prompt = True)

        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = 1024,
        )
        output = model.fast_generate(
            [text],
            sampling_params = sampling_params,
            lora_request = None,
        )[0].outputs[0].text

        print("Question:", "How many r's are in strawberry?")
        print("Answer[1]:", output)


        text = tokenizer.apply_chat_template([
            {"role" : "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content" : question},
        ], tokenize = False, add_generation_prompt = True)

        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = 1024,
        )
        output = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = model.load_lora("grpo_saved_lora"),
        )[0].outputs[0].text

        print("Answer[2]:", output)

    # Merge to 16bit
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

    # Merge to 4bit
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

    # Just LoRA adapters
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf("model", tokenizer,)
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

    # Save to 16bit GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

    # Save to q4_k_m GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model", # Change hf to your username!
            tokenizer,
            quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
            token = "",
        )
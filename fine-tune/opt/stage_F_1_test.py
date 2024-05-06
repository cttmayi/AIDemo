from colorama import init, Fore, Style
import config
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

init(autoreset=True)

inputs = [
    "What is MediaTek?",
    "What is VIVO?",
]

def generate(model_name, input):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(input, return_tensors="pt", add_special_tokens=True).to(config.device)
    output_ids = model.generate(input_ids, max_length=config.model_max_length)

    return tokenizer.decode(output_ids[0])

for name in (
        config.local_model_path,
        config.local_model_pre_path,
        config.local_model_sft_path,
        config.local_model_dpo_path,
    ):
    
    if os.path.exists(name):
        print(Fore.BLUE + f"Model: {name}")
        for input in inputs:
            print(generate(name, input))
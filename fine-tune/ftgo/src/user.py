from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.utils.dataset import create_datasets

from rouge import Rouge
from tqdm import tqdm


def create_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return model, tokenizer


input_key = "prompt"
label_key = "response"


def generate(model, tokenizer, input_str, config:GenerationConfig=None, device=None):
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, config)

    input_str = tokenizer.decode(input_ids[0])
    output_str = tokenizer.decode(output_ids[0][:-1])

    return output_str[len(input_str):]


def process(model_name_or_path, max_new_tokens=None, device=None):
    device = 'cpu' if device == 'mps' else device 
    model, tokenizer = create_model(model_name_or_path)
    model.to(device)


    config = GenerationConfig(
        # temperature=0,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    while(True):
        data = input("> ")
        output = generate(model, tokenizer, data, config=config, device=device)
        print(output)



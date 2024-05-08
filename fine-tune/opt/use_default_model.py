import config
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = config.model_name
device = config.device # 'cuda' if torch.cuda.is_available() else 'cpu

model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

input = "What is MediaTek?"
print('input: ', input)

input_tokens = tokenizer.tokenize(input, add_special_tokens=True)
print('input_tokens: ', input_tokens)

input_ids = tokenizer.encode(input, return_tensors="pt", add_special_tokens=False).to(device)
print('input_ids: ', input_ids[0])

output_ids = model.generate(input_ids, max_length=100)
print('output_ids: ', output_ids[0])
print('outputs: ', tokenizer.decode(output_ids[0]))
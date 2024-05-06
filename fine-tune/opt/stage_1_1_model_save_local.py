import config
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

model.save_pretrained(config.local_model_path)
tokenizer.save_pretrained(config.local_model_path)


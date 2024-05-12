import config

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import AutoModelForCausalLMWithValueHead

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

#pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
#    config.model_name, 
#    peft_config=lora_config,
#)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

#####################

model_name = config.model_name
model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)


model.save_pretrained(config.local_model_path)
tokenizer.save_pretrained(config.local_model_path)

#ret = model.generate(tokenizer.encode("Hello, my dog is cute", return_tensors="pt"))

#print(tokenizer.decode(ret[0]))

model = AutoModelForCausalLM.from_pretrained(config.local_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)

for name, module in model.named_modules():
    print(name)
import config
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset





model = AutoModelForCausalLM.from_pretrained(config.local_model_pre_path)
tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)
model_ref = AutoModelForCausalLM.from_pretrained(config.local_model_pre_path)


# 使用datasets库加载JSONL文件
dataset = load_dataset('json', data_files=config.local_data_dpo_path, split='train')

response_template = '### Answer:'

def process(rows):
    print(rows)
    for i in range(len(rows['prompt'])):
        rows['prompt'][i] = f"### Question: {rows['prompt'][i]}\n" + response_template  
        rows["chosen"][i] = rows["chosen"][i] + tokenizer.eos_token
        rows["rejected"][i] = rows["rejected"][i] + tokenizer.eos_token
    return rows

dataset = dataset.map(process, batched=True)

training_args = TrainingArguments(
        output_dir="./output",
        max_steps=5,
    )

trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(config.local_model_dpo_path)
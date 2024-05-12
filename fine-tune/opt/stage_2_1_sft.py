import config

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM



model = AutoModelForCausalLM.from_pretrained(config.local_model_pre_path)
tokenizer = AutoTokenizer.from_pretrained(config.local_model_pre_path)

dataset = load_dataset('json', data_files=config.local_data_sft_path, split='train')

response_template = '### Answer:'

def formatting_prompts_func(examples):
    output_texts = []
    # print(examples)
    for i in range(len(examples['prompt'])):
        text = tokenizer.bos_token + f"### Question: {examples['prompt'][i]}\n" + response_template + f"{examples['response'][i]}" + tokenizer.eos_token
        print(text)
        output_texts.append(text)
    return output_texts

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
        output_dir='./output',
        max_steps=3,
    )


trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=config.model_max_length,
    # dataset_batch_size = 2,
    # callbacks=
)

import transformers.models.opt.modeling_opt


trainer.train()

trainer.save_model(config.local_model_sft_path)
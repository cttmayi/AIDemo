# A100 18G memory
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType
import torch

model_id = 'ZhipuAI/chatglm3-6b'
model_id = 'qwen/Qwen2-0.5B'


# 拉起模型
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', trust_remote_code=True).to('cpu')
lora_config = LoraConfig(
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05)
model = Swift.prepare_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
dataset = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-en', split='train')
template = get_template(TemplateType.chatglm3, tokenizer, max_length=1024)

def encode(example):
    inst, inp, output = example['instruction'], example.get('input', None), example['output']
    if output is None:
        return {}
    if inp is None or len(inp) == 0:
        q = inst
    else:
        q = f'{inst}\n{inp}'
    example, kwargs = template.encode({'query': q, 'response': output})
    return example

dataset = dataset.map(encode).filter(lambda e: e.get('input_ids'))
dataset = dataset.train_test_split(test_size=0.001)

train_dataset, val_dataset = dataset['train'], dataset['test']


train_args = Seq2SeqTrainingArguments(
    output_dir='output',
    learning_rate=1e-4,
    num_train_epochs=2,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy='steps',
    save_strategy='steps',
    dataloader_num_workers=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer)

if __name__ == '__main__':
    trainer.train()
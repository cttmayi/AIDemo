# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_remote_path = "trl-lib/tldr"
dataset_local_path = "tldr.jsonl"
dataset_name = "tldr"
dataset_split = "train"
data_inputs = "prompt"
data_targets = "completion"

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


system_prompt = ''' conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. 
'''


def apply_chat_template(tokenizer, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors='pt').to(device)
    return model_inputs

def model_generate(model, tokenizer, prompt, max_new_tokens=128):
    model_inputs = apply_chat_template(tokenizer, prompt)
    generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 测试模型生成
    prompt = "小明的爸爸有三个儿子， 大儿子叫大毛，二儿子叫二毛，那么三儿子叫什么？"
    response = model_generate(model, tokenizer, prompt)
    print(response)

    sys.exit(0)

    if not os.path.exists(dataset_local_path):
        dataset = load_dataset(dataset_remote_path, split=dataset_split)
        print('dataset', dataset)
        dataset = dataset.filter(lambda x: len(x[data_inputs]) <= 128 and len(x[data_targets]) <= 128)
        print(dataset[0])
        dataset.to_json(dataset_local_path, orient='records', lines=True)
    else:
        dataset = load_dataset("json", data_files=dataset_local_path, split=dataset_split)

    def process_prompt(item):
        prompt = item[data_inputs]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        item[data_inputs] = text
        return item

    dataset = dataset.map(process_prompt, batched=False, num_proc=1)
    print(dataset[0])

    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO", 
        logging_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=128,
        max_completion_length=32,
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs = [
            reward_len,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()

    #测试结果
    # response = model_generate(model, tokenizer, prompt)
    # print(response)


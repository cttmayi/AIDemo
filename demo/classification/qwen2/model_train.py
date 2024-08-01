import cfg
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType


model_type = cfg.local_model_path
model_final_path = cfg.local_model_final_path
dataset_split_ratio_test = 0.1
local_dataset_path = cfg.local_dataset_path

lora_config = LoraConfig(
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05)

train_args = Seq2SeqTrainingArguments(
    # use_cpu=True,
    output_dir=cfg.OUTPUT,
    learning_rate=1e-5,
    num_train_epochs=4,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy='steps',
    save_strategy='steps',
    dataloader_num_workers=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=500,
)

if __name__ == '__main__':

    model = AutoModelForCausalLM.from_pretrained(model_type, device_map='auto')
    model = Swift.prepare_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

    template = get_template(TemplateType.default, tokenizer, max_length=1024)

    def encode(example):
        inst, output = str(example['review']), str(example['label'])
        if output is None:
            return {}
        q = inst
        example, _ = template.encode({'query': q, 'response': output})
        return example

    dataset = MsDataset.load('csv', data_files=[local_dataset_path], split='train').to_hf_dataset()
    dataset = dataset.map(encode).filter(lambda e: e.get('input_ids'))
    dataset = dataset.remove_columns(['review', 'label'])
    test_size = int(dataset.num_rows * dataset_split_ratio_test)
    if test_size < 10:
        test_size = int(dataset.num_rows * dataset_split_ratio_test * 2)
    elif test_size > 1000:
        test_size = 1000
    dataset = dataset.train_test_split(test_size=test_size, seed=0)

    train_dataset, val_dataset = dataset['train'], dataset['test']

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer)

    trainer.train()
    trainer.save_model(model_final_path)
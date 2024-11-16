import cfg
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig, PromptEncoderConfig
from swift.llm import get_template, TemplateType


model_type = cfg.local_model_path


model = AutoModelForCausalLM.from_pretrained(model_type, device_map='auto')
peft_config = LoraConfig(
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05)

peft_config = PromptEncoderConfig(
    task_type="CAUSAL_LM", num_virtual_tokens=10,
    encoder_reparameterization_type="MLP",
    encoder_dropout=0.1, encoder_hidden_size=1024)


model = Swift.prepare_model(model, peft_config)

model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)


template = get_template(TemplateType.default, tokenizer, max_length=512)

def encode(example):
    NONE = {'input_ids': [], 'labels': []}

    inp, output = example.get('query', None), example['response']
    if output is None or inp is None:
        return NONE

    example, _ = template.encode({'query': inp, 'response': output})
    # print(example)
    if example.get('input_ids') is None:
        return NONE
    return example

if __name__ == '__main__':
    # dataset = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-en', split='train')
    dataset = MsDataset.load('json', data_files=[cfg.local_dataset_path_train], split='train').to_hf_dataset()
    dataset = dataset.map(encode)
    dataset = dataset.filter(lambda e: e.get('input_ids') is not None and len(e.get('input_ids')) > 0)
    dataset = dataset.train_test_split(test_size=cfg.dataset_split_ratio_test)

    train_dataset, val_dataset = dataset['train'], dataset['test']

    print(train_dataset, val_dataset)
    print(train_dataset[0])
    print(val_dataset[0])


    train_args = Seq2SeqTrainingArguments(
        output_dir='output',
        learning_rate=1e-5,
        num_train_epochs=3,
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


    trainer.train()
    trainer.save_model(cfg.output_model_path)
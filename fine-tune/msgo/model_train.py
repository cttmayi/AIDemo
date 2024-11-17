import cfg
import libs.utils as utils
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments

if __name__ == '__main__':
    model, tokenizer, template = utils.create_model(cfg.local_model_path, cfg.template_type)

    dataset = utils.init_dataset([cfg.local_dataset_path_train], template)
    dataset = dataset.train_test_split(test_size=cfg.dataset_split_ratio_test)
    train_dataset, val_dataset = dataset['train'], dataset['test']

    train_args = Seq2SeqTrainingArguments(
        output_dir='output',
        learning_rate=1e-5,
        num_train_epochs=20,
        eval_steps=20,
        save_steps=200,
        evaluation_strategy='steps',
        save_strategy='steps',
        dataloader_num_workers=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
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
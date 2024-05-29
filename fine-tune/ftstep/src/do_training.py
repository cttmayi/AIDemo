import os
import sys


from transformers import HfArgumentParser, set_seed
from trl import SFTTrainer

from utils.model import create_model
from utils.dataset import create_datasets
from utils.argument import ModelArguments, DataTrainingArguments, TrainTrainingArguments

from utils.callback import BoardCallback, SaveCallback, NormalCallback


board_callback = BoardCallback()
save_callback = SaveCallback()
normal_callback = NormalCallback()


def main(model_args:ModelArguments, data_args:DataTrainingArguments, training_args:TrainTrainingArguments):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer, collator, dataset_preprocess= create_model(model_args)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # datasets
    train_dataset = create_datasets(
        data_args.dataset_name_or_path,
        dataset_preprocess
    )

    eval_dataset = create_datasets(
        data_args.dataset_name_or_path,
        dataset_preprocess,
        'test'
    )

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=peft_config,
        packing=data_args.packing,
        #dataset_kwargs={
        #    "append_concat_token": data_args.append_concat_token,
        #    "add_special_tokens": data_args.add_special_tokens,
        #},
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,

        # callbacks=[normal_callback],
    )
    print('-' * 40, 'Model', '-' * 40)
    trainer.accelerator.print(f"{trainer.model}")
    # print(trainer.model.config)
    print('-' * 40, 'Trainable Parameters', '-' * 40)
    try:
        trainer.model.print_trainable_parameters()
    except:
        pass

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.dtype}, {param.size()}")
    print('-' * 40, 'END', '-' * 40)

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)

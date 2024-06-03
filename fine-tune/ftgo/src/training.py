
from transformers import set_seed
from transformers import DataCollatorForSeq2Seq

from src.utils.trainer import SFTTrainer
from src.utils.model import create_model
from src.utils.dataset import create_datasets
from src.default import BasicArguments, TrainArguments
# from src.utils.callback import BoardCallback


def process(basic_args:BasicArguments, training_args:TrainArguments):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_model(basic_args)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing
    # if training_args.gradient_checkpointing:
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": basic_args.use_reentrant}

    # datasets
    train_dataset = create_datasets(basic_args.dataset_name_or_path,)
    eval_dataset = create_datasets(basic_args.dataset_name_or_path,'test')

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=False, # dataset_args.packing,
        dataset_text_field= 'text', # dataset_args.dataset_text_field,
        max_seq_length=basic_args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        # callbacks=[BoardCallback()],
    )
    print('-' * 40, 'Model', '-' * 40)
    trainer.accelerator.print(f"{trainer.model}")
    # print(trainer.model.config)
    print('-' * 40, 'Trainable Parameters', '-' * 40)
    try:
        trainer.model.print_trainable_parameters()
    except:
        pass

    #for name, param in trainer.model.named_parameters():
    #    if param.requires_grad:
    #        print(f"{name}: {param.dtype}, {param.size()}")
    print('-' * 40, 'END', '-' * 40)

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir=training_args.model_output_dir)
import src.utils.env
import os
import matplotlib.pyplot as plt
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Qwen2TokenizerFast, Qwen2ForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint

from src.dataset import create_dataset
from src.model import create_model


train_dataset_path = "data/android/1M.log_structured.csv"
eval_dataset_path = "data/android/2k.log_structured.csv"
model_path = "models/Qwen2.5-0.5B-Instruct"
output_path = "output"

checkpoint = None if not os.path.exists(output_path) else get_last_checkpoint(output_path)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=100_000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=1_000,  # 保存中间模型
    save_total_limit=10,
    use_cpu=True,
    # bf16=True,
    # save_only_model=True,
    logging_steps=20,
)



if __name__ == '__main__':
    train_dataset =  create_dataset(train_dataset_path)
    eval_dataset = create_dataset(eval_dataset_path)
    model = create_model(model_path)

    # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # 保存模型
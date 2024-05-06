import config
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(config.local_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)

# 使用datasets库加载JSONL文件
dataset = load_dataset('json', data_files=config.local_data_pre_path, split='train')

# 打印第一个样本的内容
print(dataset[0])

def preprocess_function(examples):
    # 在这里实现你的数据预处理逻辑
    # 例如，你可能需要对文本进行分词处理
    # examples是一个字典，包含了一个样本的所有字段
    return examples

# 应用预处理函数
dataset = dataset.map(preprocess_function)

training_args = TrainingArguments(
        output_dir='./output',
        max_steps=10,
    )

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config.model_max_length,
)

trainer.train()

trainer.save_model(config.local_model_pre_path)
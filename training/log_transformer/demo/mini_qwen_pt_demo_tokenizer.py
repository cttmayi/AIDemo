import os, sys
import torch
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
)

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载分词器与模型
model_path = "models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)

print(tokenizer)
# print(tokenizer.vocab)

# 保存当前词汇表文件
vocab_file = tokenizer.save_vocabulary(save_directory="temp")
print(vocab_file)





text = "PID 2 MODE 0 TE TE"

# 将文本编码为token
tokens = tokenizer.encode(text)

print(tokens)

# 将token解码为文本
decoded_text = tokenizer.decode(tokens)

print(decoded_text)


sys.exit()


# 准备新的词汇表
new_vocab = ["TE", "MODE", 'PID']  # 替换为完整的词汇表

# 替换词汇表
tokenizer.vocab = {token: idx for idx, token in enumerate(new_vocab)}
tokenizer.tokens_to_ids = {token: idx for idx, token in enumerate(new_vocab)}

# 将文本编码为token
tokens = tokenizer.encode(text)

print(tokens)

# 将token解码为文本
decoded_text = tokenizer.decode(tokens)

print(decoded_text)



sys.exit() 



# 保存自定义分词器
tokenizer.save_pretrained("path_to_custom_tokenizer")

# 重新加载自定义分词器
custom_tokenizer = AutoTokenizer.from_pretrained("path_to_custom_tokenizer")
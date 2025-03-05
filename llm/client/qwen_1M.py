# https://qwenlm.github.io/zh/blog/qwen2.5-turbo/

import os
from utils import env 
env.chdir(__file__)


from openai import OpenAI

# 读取长文本文件
with open("example.txt", "r", encoding="utf-8") as f:
    text = f.read()
user_input = text + "\n\nSummarize the above text."

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],   # https://bailian.console.aliyun.com/?apiKey=1#/api-key
    base_url=os.environ["DASHSCOPE_API_BASE"], # https://dashscope.aliyuncs.com/compatible-mode/v1
)

completion = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
      {'role': 'system', 'content': 'You are a helpful assistant.'},
      {'role': 'user', 'content': user_input},
    ],
)

print(completion.choices[0].message)



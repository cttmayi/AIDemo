import os
from swift.llm import TemplateType

PROJECT = 'msgo'

model_name = 'qwen/Qwen2-0.5B'
dataset_name = 'wowhaha/moral-foundation-news_2000'# 'AI-ModelScope/alpaca-gpt4-data-en' # "afqmc"
template_type = TemplateType.default

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# CACHE = os.path.join(CURR_PATH, "local_cache")
CACHE = "cache"
PROJECT_CACHE = os.path.join(CACHE, PROJECT)


local_model_path = os.path.join(PROJECT_CACHE, 'model', model_name.split('/')[-1])
dataset_split_ratio_test = 0.1

local_dataset_path_train = os.path.join(PROJECT_CACHE, 'dataset', 'train.json')
# local_dataset_path_eval = os.path.join(MSGO_CACHE, 'dataset', 'eval.json')
local_dataset_path_test = os.path.join(PROJECT_CACHE, 'dataset', 'test.json')

output_model_path = os.path.join(PROJECT_CACHE, 'output')


for path in [CACHE, PROJECT_CACHE, output_model_path]:
    if path and not os.path.exists(path):
        os.makedirs(path)
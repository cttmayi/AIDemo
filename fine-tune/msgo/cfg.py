import os
from swift.llm import TemplateType




model_name = 'Qwen/Qwen3-0.6B-Base'
dataset_name = 'wowhaha/moral-foundation-news_2000'# 'AI-ModelScope/alpaca-gpt4-data-en' # "afqmc"
template_type = TemplateType.default

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.basename(CURR_PATH)

CACHE = ".cache"
PROJECT_CACHE = os.path.join(CACHE, PROJECT)


local_model_path = os.path.join(CACHE, 'model', model_name.split('/')[-1])
dataset_split_ratio_test = 0.1

local_dataset_path_train = os.path.join(CACHE, 'dataset', 'train.json')
# local_dataset_path_eval = os.path.join(CACHE, 'dataset', 'eval.json')
local_dataset_path_test = os.path.join(CACHE, 'dataset', 'test.json')

output_model_path = os.path.join(PROJECT_CACHE, 'output')

for path in [CACHE, PROJECT_CACHE, output_model_path]:
    if path and not os.path.exists(path):
        os.makedirs(path)
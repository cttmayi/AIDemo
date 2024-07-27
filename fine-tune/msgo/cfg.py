import os

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# CACHE = os.path.join(CURR_PATH, "local_cache")
CACHE = "cache"
MSGO_CACHE = os.path.join(CACHE, "msgo")
for path in [CACHE, MSGO_CACHE]:
    if path and not os.path.exists(path):
        os.makedirs(path)



model_name = 'qwen/Qwen2-0.5B'
local_model_path = os.path.join(MSGO_CACHE, 'model', model_name.split('/')[-1])

dataset_name = 'AI-ModelScope/alpaca-gpt4-data-en' # "afqmc"
dataset_split_ratio_test = 0.1


local_dataset_path_train = os.path.join(MSGO_CACHE, 'dataset', 'train.json')
# local_dataset_path_eval = os.path.join(MSGO_CACHE, 'dataset', 'eval.json')
local_dataset_path_test = os.path.join(MSGO_CACHE, 'dataset', 'test.json')


import os

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# CACHE = os.path.join(CURR_PATH, "local_cache")
OUTPUT = "output"
CACHE = "cache"
MSGO_CACHE = os.path.join(CACHE, "msgo")
for path in [CACHE, MSGO_CACHE]:
    if path and not os.path.exists(path):
        os.makedirs(path)


model_name = 'qwen/Qwen2-0.5B'
local_model_path = os.path.join(MSGO_CACHE, 'model', model_name.split('/')[-1])

local_model_final_path = os.path.join(MSGO_CACHE, 'model', 'final')

local_dataset_path = 'data/waimai/waimai_10k_train.csv'
dataset_split_ratio_test = 0.05

local_dataset_test_path = 'data/waimai/waimai_10k_test.csv'

import torch

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model_name = 'facebook/opt-350m'

local_model_path = 'models/opt.pt'
local_model_pre_path = 'models/opt_pre.pt'
local_model_sft_path = 'models/opt_sft.pt'
local_model_dpo_path = 'models/opt_dpo.pt'

model_max_length = 128

local_data_pre_path = 'data/pretrain_data.jsonl'
local_data_sft_path = 'data/sft_data.jsonl'
local_data_dpo_path = 'data/dpo_data.jsonl'
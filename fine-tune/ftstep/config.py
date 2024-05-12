import torch
import os


if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

curdir = os.path.normpath(os.path.dirname(__file__))
print(curdir)


model_name = 'facebook/opt-350m'

local_model_path = os.path.join(curdir, 'models', 'opt_base.pt')
local_model_pre_path = os.path.join(curdir, 'models', 'opt_pre.pt')
local_model_sft_path = os.path.join(curdir, 'models', 'opt_sft.pt')
local_model_dpo_path = os.path.join(curdir, 'models', 'opt_dpo.pt')

model_max_length = 128

local_data_pre_path = os.path.join(curdir, 'data', 'pretrain_data.jsonl')
local_data_sft_path = os.path.join(curdir, 'data', 'sft_data.jsonl')
local_data_dpo_path = os.path.join(curdir, 'data', 'dpo_data.jsonl')

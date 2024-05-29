import torch
import os


if torch.backends.mps.is_available():
    default_device = 'mps'
elif torch.cuda.is_available():
    default_device = 'cuda'
else:
    default_device = 'cpu'

curdir = os.path.normpath(os.path.dirname(__file__))

default_model_name = 'facebook/opt-125m'
defualt_model_max_length = 128


default_remote_dataset_name = None
default_local_dataset_path = None









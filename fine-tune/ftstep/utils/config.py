import torch
import os


if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

curdir = os.path.normpath(os.path.dirname(__file__))
# print(curdir)

default_model_name = 'facebook/opt-125m'
defualt_model_max_length = 128



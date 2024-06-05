import torch
import os


if torch.backends.mps.is_available():
    default_device = 'mps'
elif torch.cuda.is_available():
    default_device = 'cuda'
else:
    default_device = 'cpu'










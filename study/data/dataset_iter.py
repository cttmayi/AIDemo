import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm

data_path = '12345678901234567890'

# step:1
class IterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path # Only load file path

    def __iter__(self):
         # Process a single data depending on how the data is read
         for item in self.file_path:
             yield item

# step：2
dataset = IterableDataset(data_path)

# step:3
dataloader = DataLoader(
        dataset,
        batch_size=4,
    )
    

if __name__ == '__main__':
    # dataloader = tqdm(dataloader)
    for idx, batch in enumerate(dataloader):
        print(idx, batch)
    pass
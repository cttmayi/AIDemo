import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import os
import random


from torch.utils.data import Dataset, IterableDataset, DataLoader

train_data_size = 1000
test_data_size = 10



class RegDataset(Dataset):
    def __init__(self, data_path):

        self.regs = []
        self.labels = []
        len_data = len(self._data(0))

        if data_path == 'train':
            for i in range(train_data_size):
                self.regs.append(self._data(i))
                self.labels.append(-1)
        elif data_path == 'test':
            for i in range(test_data_size):
                err_i = i # random.randint(0, len_data - 1)
                index = random.randint(0, train_data_size - 1)
                if i < 2 or err_i < 2 :
                    self.regs.append(self._data(index))
                    self.labels.append(-1)
                else:
                    self.regs.append(self._data_err(index, err_i))
                    self.labels.append(err_i)

        pass

    def _data(self, index):
        i = index % 100
        g = int(index / 100)
        data = [
            i, 
            g,
            i * 2,
            i + g,
            i + 1, 
            int(i/2),
            i + 2, 
            7, 
            i + 3, 
            random.randint(0, 5)
        ]
        return data

    def _data_err(self, index, err_index):

        err_range = [
            [1, 3], # 0
            [1, 3], # 1
            [1, 3], # 2
            [1, 3], # 3
            [1, 3], # 4
            [1, 3], # 5
            [1, 3], # 6
            [1, 3], # 7
            [1, 3], # 8
            [10, 20], # 9
        ]

        data = self._data(index)
        
        data[err_index] = data[err_index] + random.randint(err_range[err_index][0], err_range[err_index][1])
        return data

    def __getitem__(self, index):
        reg = torch.tensor(self.regs[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return reg, label

    def __len__(self):
        # 返回数据集的大小
        return len(self.regs)


def create_dataset(data_path, batch_size):
    train_dataset = RegDataset('train')
    test_dataset = RegDataset('test')

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )    

    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            # shuffle=True,
        ) 

    return train_loader, test_loader


if __name__ == '__main__':
    device = 'mps'

    train_loader, test_loader = create_dataset('', 2)

    for idx, batch in enumerate(train_loader):
        print(idx, batch)
        # batch = [x.to(device) for x in batch] # to device

    #for d, l in train_loader:
    #    print(d, l)
    #    pass

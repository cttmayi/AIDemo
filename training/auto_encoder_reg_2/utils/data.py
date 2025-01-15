import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import os
import random


from torch.utils.data import Dataset, IterableDataset, DataLoader

train_data_size = 1000
test_data_size = 21


def _d_R(index):
    index = index % 3
    R = [ 
        [ 480,  720,  480 + 20, 10],
        [ 720, 1280,  720 + 35, 15],
        [1080, 1920, 1080 + 45, 20],
    ]
    return R[index]

def _d_M(index):
    index = index % 2
    M = [ 
        [ 1,  3,  5,  2,  0],
        [ 0,  3,  3,  4,  1],
    ]
    return M[index]

def _d_F(index):
    return [4, 7, 10, 2, 1]


class RegDataset(Dataset):
    def __init__(self, data_path):

        self.regs = []
        self.labels = []
        self.len_data = len(self._data(0))

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
                    self.regs.append(self._data(index))
                    self.labels.append(-1)
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
            i // 2,
            random.randint(0, 5)
        ]
        R = _d_R(index)
        M  = _d_M(index)
        F  = _d_F(index)
        data.extend(R)
        data.extend(M)
        data.extend(F)
        return data

    def _data_err(self, index, err_index):
        data = self._data(index)
        data[err_index] = data[err_index] + random.randint(1, 3)
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

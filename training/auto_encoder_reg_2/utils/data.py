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


# To make the variable and function names more descriptive and meaningful, I've rewritten the code with improved naming conventions. The changes include making the names more specific to their purpose and adhering to Python's naming conventions (snake_case for variables and functions).

def get_modulation_matrix_5x5(index):
    index = index % 5
    modulation_matrix_5x5 = [ 
        [1, 3, 5, 2, 0],
        [0, 3, 3, 4, 1],
        [5, 2, 3, 4, 1],
        [1, 2, 3, 4, 1],
        [0, 3, 3, 4, 1],
    ]
    return modulation_matrix_5x5[index]

def get_modulation_matrix_6x5(index):
    index = index % 6
    modulation_matrix_6x5 = [ 
        [1, 3, 5, 2, 0],
        [0, 3, 3, 4, 1],
        [2, 2, 3, 4, 1],
        [1, 2, 3, 4, 1],
        [0, 3, 3, 4, 1],
        [2, 2, 3, 4, 1],
    ]
    return modulation_matrix_6x5[index]

def get_frequency_vector(index):
    return [4, 7, 1, 2, 1]

def generate_data(index, error_index=None):
    data = []
    
    modulation_vector_5x5 = get_modulation_matrix_5x5(index)
    modulation_vector_6x5 = get_modulation_matrix_6x5(index)
    frequency_vector = get_frequency_vector(index)
    
    data.extend(modulation_vector_5x5)
    data.extend(modulation_vector_6x5)
    data.extend(frequency_vector)

    # data = [value * 100 for value in data]

    if error_index is not None:
        error_index = error_index % len(data)
        data[error_index] = data[error_index] + random.randint(1, 3)
    data = number_to_binary_lists(data, length=12)

    return data

def number_to_binary_list(n, length=4):
    binary_str = bin(n)[2:]
    binary_str = binary_str.zfill(length)
    binary_list = [int(bit) for bit in binary_str]
    return binary_list

def number_to_binary_lists(numbers, length=4):
    binary_lists = [number_to_binary_list(n, length) for n in numbers]
    return [item for sublist in binary_lists for item in sublist]


data_len = len(generate_data(0))

class RegDataset(Dataset):
    def __init__(self, data_path, encoding='label'):

        self.data = []
        self.data_err = []
        self.labels = []
        self.encoding = encoding

        if data_path == 'train':
            for i in range(train_data_size):
                self.data.append(generate_data(i))
                self.data_err.append(generate_data(i, i))
        elif data_path == 'test':
            for i in range(test_data_size):
                err_i = i # random.randint(0, len_data - 1)
                index = random.randint(0, train_data_size - 1)

                self.data.append(generate_data(index))
                self.data_err.append(generate_data(index, err_i))


    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        data_err = torch.tensor(self.data_err[index], dtype=torch.float32)
        return data, data_err

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)


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
    # device = 'mps'

    train_loader, test_loader = create_dataset('', 2)

    for idx, batch in enumerate(train_loader):
        print(idx, batch)
        # batch = [x.to(device) for x in batch] # to device

    #for d, l in train_loader:
    #    print(d, l)
    #    pass

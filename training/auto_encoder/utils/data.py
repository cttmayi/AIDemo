import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import os





def create_dataset(dataset_name, data_dir, batch_size):
    train_loader = test_loader = None

    if dataset_name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
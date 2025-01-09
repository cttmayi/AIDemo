import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm

data_path = '12345678901234567890'

# step:1
class MapDataset(Dataset): # 需要继承Dataset类
    def __init__(self, data_path):
        # 类变量初始化的地方
        self.data_path = data_path
        pass

    def __getitem__(self, index):
        # 用来定义数据从DataLoader中的取出规则
        # 可以在这一部分把图片数据格式转化为tensor
        # 如果数据不是随机取，那么index 是从0开始，取决于DataLoader中的shuffle参数。
        datas = {'index': index}
        datas = index
        targets = 0
        return datas, targets # 返回你所需要的数据

    def __len__(self):
        # 返回数据集的大小
        return 10

# step：2
dataset = MapDataset(data_path)

# step:3
dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )
    

if __name__ == '__main__':
    device = 'mps'
    # dataloader = tqdm(dataloader)
    for idx, batch in enumerate(dataloader):
        print(idx, batch)
        batch = [x.to(device) for x in batch] # to device

    for d, t in dataloader:
        print(d, t)
        pass
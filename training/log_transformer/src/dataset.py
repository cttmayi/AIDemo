

# pytorch dataset for log transformer
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py

from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import random
import csv
import numpy as np
from transformers import AutoTokenizer

FILE = 'data/Android_2k.log_structured.csv'


_str_dict = {}
def str2int(s):
    if s not in _str_dict:
        _str_dict[s] = len(_str_dict)
    return _str_dict[s]

class LogDataset(Dataset):
    def __init__(self, data_path,):
        self.data = []

        # 打开CSV文件
        with open(data_path, mode='r', encoding='utf-8') as file:
            # 创建一个csv读取器
            csv_reader = csv.reader(file)
            
            one_data = []
            for l, row in enumerate(csv_reader):
                if l == 0:
                    continue
                # ['1506', '03-17', '16:15:49.587', '2227', '2227', 'V', 'PhoneStatusBar', 'setLightsOn(true)', 'E123', 'setLightsOn(true)']
                one_data.append(str2int(row[4]))
                one_data.append(str2int(row[8]))
                for v in row[9]:
                    if v == '*':
                        one_data.append(str2int(v))
                if len(one_data) > 100:
                    self.data.append(one_data)
                    one_data = []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        result = {}
        result['input_ids'] = torch.tensor(item, dtype=torch.long)
        result['attention_mask'] = torch.ones_like(result['input_ids'])
        result['labels'] = torch.tensor(item, dtype=torch.long)
        return result



import torch

# 假设我们有两个字典，分别代表选择（choice）和拒绝（reject）的输入ID张量
choice = {
    'input_ids': torch.tensor([1, 2, 3, 4, 5])  # 假设的输入ID序列
}

reject = {
    'input_ids': torch.tensor([6, 7, 8, 9, 10])  # 假设的输入ID序列
}

# 使用torch.cat将两个输入ID张量按第一个维度（dim=0）拼接起来
input_ids = torch.cat([choice['input_ids'], reject['input_ids']], dim=0)

# 打印结果
print("Concatenated input IDs: ", input_ids)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设我们有一个模型的输出logits，形状为[batch_size, num_classes]
logits = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

# 假设这是正确的类别标签
targets = torch.tensor([0, 2])

# 创建CrossEntropyLoss实例
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(logits, targets)

print(loss)  # 输出损失值
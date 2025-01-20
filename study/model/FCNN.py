import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 全连接神经网络（Fully Connected Neural Network, FCNN）

class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 示例
input_dim = 784  # 28x28图像展平后的维度
hidden_dim = 128
output_dim = 10  # 10个类别

model = FCNN(input_dim, hidden_dim, output_dim)
x = torch.randn(1, input_dim)  # 生成一个随机输入
output = model(x)
print("输入：", x)
print("输出：", output)
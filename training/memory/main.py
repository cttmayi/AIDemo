
# 线性模型训练代码
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

DEVICE = 'mps'
DTYPE = torch.float32
B = 1
NX = 2148
NY = int(1220 * 10 * 1.5)

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(NX, NY // 2, dtype=DTYPE)
        self.linear2 = nn.Linear(NY // 2, NY, dtype=DTYPE)
    def forward(self, x):
        # x = self.linear1(x)
        x = checkpoint(self.linear1, x)
        x = self.linear2(x)
        # x = checkpoint(self.linear2, x)
        return x

# 创建模型实例
model = SimpleModel().to(DEVICE)
print(model)
# 打印模型参数数量
print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024, 'M')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# 定义输入和目标数据
inputs = torch.randn(B, NX, dtype=DTYPE).to(DEVICE)
targets = torch.randn(B, NY, dtype=DTYPE).to(DEVICE)
# inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.float32).repeat(32, 1).to("mps")
# targets = torch.tensor([[50.0]], dtype=torch.float32).repeat(32, 1).to("mps")

import time
start = time.time()
delay = 10
# 训练循环
for epoch in range(10000):

    # 前向传播

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # 反向传播和优化
    loss.backward()
    optimizer.step()


    # 清零梯度
    optimizer.zero_grad()
    end = time.time()
    if end - start > delay:
    # 打印损失
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
        print('Time: %.2f' % ((end - start)/(epoch + 1)))
        delay += 10



# Memory
# 211MS
# 220MS

# NX = 2150, NY = 1000 ADAM  
# 0.0025B = 0.002*1024*1024 = 
# 322MS


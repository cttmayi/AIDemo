import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

torch.mps.manual_seed(0)  # 设置随机种子以确保可重复性

# 检查是否支持 MPS
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available. Please check your PyTorch version and hardware.")

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型、损失函数和优化器
model = SimpleModel().to("mps")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建 GradScaler 对象
scaler = GradScaler()

# 模拟训练数据
inputs = torch.randn(32, 10).to("mps")
targets = torch.randn(32, 1).to("mps")

inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.float32).repeat(32, 1).to("mps")
targets = torch.tensor([[50.0]], dtype=torch.float32).repeat(32, 1).to("mps")

# 训练循环
for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    
    # 使用 autocast 进行前向传播
    with autocast(device_type="mps", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 使用 GradScaler 进行反向传播和优化
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if epoch % 1000 == 0:
        # 打印损失
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
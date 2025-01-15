import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义神经网络模型
class CompareNet(nn.Module):
    def __init__(self):
        super(CompareNet, self).__init__()
        self.fcr1 = nn.Linear(2, 16)  # 输入层，2个输入特征，16个神经元
        self.relu = nn.ReLU()
        self.fcr2 = nn.Linear(16, 8)  # 隐藏层，8个神经元
        # self.fcr3 = nn.Linear(8, 1)   # 输出层，1个神经元

        self.fc1 = nn.Linear(2, 16)  # 输入层，2个输入特征，16个神经元
        self.fc2 = nn.Linear(16, 8)  # 隐藏层，8个神经元
        
        
        self.fc3 = nn.Linear(8, 1)   # 输出层，1个神经元


    def forward(self, x):
        x2 = x.clone()
        
        x = self.relu(self.fcr1(x))
        x = self.relu(self.fcr2(x))
        # x = torch.sigmoid(self.fcr3(x))
        
        #x2 = self.fc1(x2)
        #x2 = self.fc2(x2)

        # x = self.fc3(torch.cat((x, x2), dim=1))
        x = self.fc3(x)
                     
        return x

# 生成训练数据
num_samples = 10000
x_train = np.random.rand(num_samples, 2).astype(np.float32)  # 生成10000个样本，每个样本包含两个随机数字
y_train = np.zeros(num_samples).astype(np.float32)
for i in range(num_samples):
    #if x_train[i, 0] > x_train[i, 1]:
    #    y_train[i] = 1  # 如果第一个数字大于第二个数字，标签为1，否则为0
    y_train[i] = x_train[i, 0] * x_train[i, 1]  # 训练数据为两个数字的差值

# 转换为PyTorch张量
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).unsqueeze(1)  # 增加一个维度，使其成为二维张量

# 初始化模型、损失函数和优化器
model = CompareNet()
# criterion = nn.BCELoss()  # 二元交叉熵损失函数
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 100
batch_size = 128
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        inputs = x_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.9f}')

# 测试模型
x_test = torch.tensor([[0.5, 0.3], [0.2, 0.7], [90, 1], [4, 6]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_test = model(x_test)
print("测试数据:")
print("输入数字对:", x_test)
print("预测结果:", y_test)
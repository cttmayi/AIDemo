import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 示例：生成模拟多维状态变化日志
# 假设每个状态是一个多维向量，状态变化是从一个状态向量到另一个状态向量的转移
def generate_normal_state_transitions(num_samples, state_dim):
    # 随机生成正常的状态变化
    normal_transitions = np.random.normal(loc=0.5, scale=0.1, size=(num_samples, state_dim * 2))
    return torch.tensor(normal_transitions, dtype=torch.float32)

def generate_anomaly_state_transitions(num_samples, state_dim):
    # 生成异常的状态变化（与正常数据有明显差异）
    anomaly_transitions = np.random.normal(loc=0.8, scale=0.2, size=(num_samples, state_dim * 2))
    return torch.tensor(anomaly_transitions, dtype=torch.float32)

# 定义 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 训练 Autoencoder
def train_autoencoder(model, train_loader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, _ = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 检测异常状态变化
def detect_anomalies(model, data, threshold):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        reconstruction_error = torch.mean((reconstructed - data) ** 2, dim=1)
        anomalies = reconstruction_error > threshold
    return anomalies

# 参数设置
state_dim = 5  # 假设每个状态是一个5维向量
input_dim = state_dim * 2  # 每个状态变化记录两个状态（起始状态和目标状态）
hidden_dim = 16
num_normal_samples = 1000
num_anomaly_samples = 100
batch_size = 64
epochs = 50
learning_rate = 0.001
anomaly_threshold = 0.05  # 根据需要调整阈值

# 生成正常状态变化日志
normal_transitions = generate_normal_state_transitions(num_normal_samples, state_dim)
print("Normal Transitions Shape:", normal_transitions.shape)
print("Normal Transitions:", normal_transitions)

train_dataset = TensorDataset(normal_transitions, normal_transitions)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化并训练 Autoencoder
model = Autoencoder(input_dim, hidden_dim)
train_autoencoder(model, train_loader, epochs, learning_rate)

# 生成异常状态变化日志
anomaly_transitions = generate_anomaly_state_transitions(num_anomaly_samples, state_dim)

# 检测异常
all_data = torch.cat((normal_transitions, anomaly_transitions), dim=0)
anomalies = detect_anomalies(model, all_data, anomaly_threshold)

# 输出结果
print("Anomalies detected:", anomalies)
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np

# 1. 构建图数据
# 节点特征
node_features = torch.tensor([
    [5.0, 3.0, 4.0],  # Running
    [5.0, 2.0, 2.0],  # Paused
    [5.0, 1.0, 1.0]   # Restarting
], dtype=torch.float)

# 边索引（无向图）
edge_index = torch.tensor([
    [0, 1, 0, 2],  # Running -> Paused, Running -> Restarting
    [1, 0, 2, 0]   # Paused -> Running, Restarting -> Running
], dtype=torch.long)

# 构建图数据对象
data = Data(x=node_features, edge_index=edge_index)

# 2. 定义GNN Autoencoder模型
class GNN_Autoencoder(torch.nn.Module):
    def __init__(self):
        super(GNN_Autoencoder, self).__init__()
        print('feateures:', data.num_features)
        self.encoder = GCNConv(data.num_features, 16)
        self.decoder = GCNConv(16, data.num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        reconstructed_x = self.decoder(z, edge_index)
        return reconstructed_x

# 3. 训练模型
model = GNN_Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    reconstructed_x = model(data)
    loss = F.mse_loss(reconstructed_x, data.x)  # 重构损失
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# 4. 检测异常
model.eval()
with torch.no_grad():
    reconstructed_x = model(data)
    reconstruction_error = torch.norm(reconstructed_x - data.x, dim=1)
    print("重构的节点特征:")
    print(reconstructed_x)
    print("原始节点特征:")
    print(data.x)
    print("重构误差:")
    print(reconstruction_error)

    print(f'Reconstruction Errors: {reconstruction_error}')




# 计算边的重构误差
def calculate_edge_reconstruction_error(data, reconstructed_x):
    # 使用重构的节点特征计算边的重构误差
    # 这里假设边的权重是节点特征的某种函数
    edge_index = data.edge_index
    src_features = reconstructed_x[edge_index[0]]
    dst_features = reconstructed_x[edge_index[1]]
    reconstructed_edge_weights = torch.norm(src_features - dst_features, dim=1)
    print(f'-- Reconstructed Edge Weights: {reconstructed_edge_weights}')
    
    # 假设原始边权重为1（无权重图）
    original_edge_weights = torch.ones(edge_index.shape[1])
    print(f'-- Original Edge Weights: {original_edge_weights}')
    edge_reconstruction_error = torch.norm(reconstructed_edge_weights - original_edge_weights, dim=0)
    return edge_reconstruction_error

# 检测异常边
edge_reconstruction_error = calculate_edge_reconstruction_error(data, reconstructed_x)
print(f'Edge Reconstruction Errors: {edge_reconstruction_error}')

# 设置边异常阈值
edge_threshold = np.percentile(edge_reconstruction_error.numpy(), 99)  # 选择99%的百分位数作为阈值
print(f'Edge Threshold: {edge_threshold}')

# 分析异常边
# for i in range(edge_reconstruction_error.shape[0]):
#     if edge_reconstruction_error[i] > edge_threshold:
#         #src_state = logs[edge_index[0, i]][1]
#         #dst_state = logs[edge_index[1, i]][1]
#         #print(f"异常边：从 {src_state} 到 {dst_state}，重构误差：{edge_reconstruction_error[i].item()}")
#         print(f"异常边 重构误差：{edge_reconstruction_error[i].item()}")
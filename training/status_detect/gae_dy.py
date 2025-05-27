import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 1. 动态构建图数据
def build_graph_from_logs(logs):
    # 假设logs是一个列表，每个元素是一个包含时间戳、状态标识和状态描述的元组
    # logs = [(timestamp, state, description), ...]
    
    # 提取唯一状态
    unique_states = set(state for _, state, _ in logs)
    state_to_index = {state: idx for idx, state in enumerate(unique_states)}
    
    # 构建节点特征（这里简单使用状态出现次数作为特征）
    node_features = torch.zeros(len(unique_states), 1)
    for _, state, _ in logs:
        node_features[state_to_index[state]] += 1
    
    # 构建边索引（状态转换）
    edge_index = []
    for i in range(len(logs) - 1):
        src_state = logs[i][1]
        dst_state = logs[i + 1][1]
        edge_index.append([state_to_index[src_state], state_to_index[dst_state]])
        # edge_index.append([state_to_index[dst_state], state_to_index[src_state]])  # 无向图
    #print(torch.tensor(edge_index, dtype=torch.long))
    #print(torch.tensor(edge_index, dtype=torch.long).t())
    #print(torch.tensor(edge_index, dtype=torch.long).t().contiguous())
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=node_features, edge_index=edge_index)

# 示例日志数据
logs = [
    ("2025-05-23 00:00:00", "Running", "设备正常运行"),
    ("2025-05-23 00:05:00", "Paused", "设备暂停"),
    ("2025-05-23 00:10:00", "Running", "设备恢复运行"),
    ("2025-05-23 00:15:00", "Restarting", "设备重启"),
    ("2025-05-23 00:20:00", "Running", "设备正常运行"),
]

data = build_graph_from_logs(logs)

print("构建的图数据:")
print(data.x) # 节点特征
print(data.edge_index) # 边索引

# 2. 定义GNN Autoencoder模型
class GNN_Autoencoder(torch.nn.Module):
    def __init__(self, num_features):
        super(GNN_Autoencoder, self).__init__()
        self.encoder = GCNConv(num_features, 16)
        self.decoder = GCNConv(16, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        reconstructed_x = self.decoder(z, edge_index)
        return reconstructed_x

# 3. 训练模型
model = GNN_Autoencoder(num_features=data.num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
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
    print("重构的节点特征:")
    print(reconstructed_x)
    reconstruction_error = torch.norm(reconstructed_x - data.x, dim=1)
    print(f'Reconstruction Errors: {reconstruction_error}')
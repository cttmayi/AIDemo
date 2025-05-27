import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

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

# 2. 定义GNN模型
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, 1)  # 输出异常分数

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # 聚合所有节点的特征
        x = self.fc(x)
        return x

# 3. 训练模型
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 模拟训练过程（这里使用随机生成的标签作为示例）
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, torch.tensor([0.0]))  # 假设正常状态的异常分数为0
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# 4. 检测异常
model.eval()
with torch.no_grad():
    anomaly_score = model(data)
    print(f'Anomaly Score: {anomaly_score.item()}')
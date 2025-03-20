import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        indices = torch.tensor(item[0], dtype=torch.long)  # 确保索引是 LongTensor
        features = torch.tensor(item[1:], dtype=torch.float32)  # 其他特征可以是 FloatTensor
        return torch.cat((indices.unsqueeze(0), features), dim=0)

# 定义模型
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        indices = x[:, 0].long()  # 确保索引部分是 LongTensor
        embeddings = self.embedding(indices)
        x = self.transformer_encoder(embeddings.permute(1, 0, 2))
        return self.fc(x.permute(1, 0, 2))

# 数据
data = [[1, 2.0, 3.0], [4, 5.0, 6.0], [7, 8.0, 9.0]]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 模型和优化器
model = TransformerEncoderModel(vocab_size=10, embedding_dim=16, d_model=32, num_heads=4, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
for epoch in range(10):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch[:, 0].long())  # 确保目标也是 LongTensor
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/10], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")
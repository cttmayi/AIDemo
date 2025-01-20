import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 取平均值
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 示例
input_dim = 10000  # 词汇表大小
num_heads = 8
hidden_dim = 512
output_dim = 10  # 10个类别
num_layers = 6
seq_length = 5  # 序列长度

model = TransformerModel(input_dim, num_heads, hidden_dim, output_dim, num_layers)
x = torch.randint(0, input_dim, (seq_length, 1))  # 生成一个随机输入
output = model(x)
print("输入：", x)
print("输出：", output)
import torch
import torch.nn as nn

# 长短期记忆网络（Long Short-Term Memory, LSTM）

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 示例
input_dim = 10  # 每个时间步的输入维度
hidden_dim = 128
output_dim = 10  # 10个类别
num_layers = 2
seq_length = 5  # 序列长度

model = LSTM(input_dim, hidden_dim, output_dim, num_layers)
x = torch.randn(1, seq_length, input_dim)  # 生成一个随机输入
output = model(x)
print("输入：", x)
print("输出：", output)
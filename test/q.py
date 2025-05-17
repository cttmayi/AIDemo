import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的线性网络
class SimpleLinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 设置输入维度和输出维度
input_dim = 10
output_dim = 1

# 创建网络实例
net = SimpleLinearNet(input_dim, output_dim)

# 打印原始网络参数
print("原始网络参数:")
for name, param in net.named_parameters():
    print(f"{name}: {param}")

# 量化参数 w
def quantize_weights(weights, bits=8):
    """
    量化权重函数
    :param weights: 要量化的权重
    :param bits: 量化位数
    :return: 量化后的权重
    """
    # 获取量化范围
    q_min = -2 ** (bits - 1)
    q_max = 2 ** (bits - 1) - 1

    # 计算量化比例因子
    scale = (q_max - q_min) / (weights.max() - weights.min())

    # 量化权重
    quantized_weights = torch.round(weights * scale) / scale

    return quantized_weights

# 量化网络参数
with torch.no_grad():
    for name, param in net.named_parameters():
        if "weight" in name:
            quantized_weights = quantize_weights(param)
            # param.copy_(quantized_weights)

# 打印量化后的网络参数
print("\n量化后的网络参数:")
for name, param in net.named_parameters():
    print(f"{name}: {param}")

# 计算量化误差
quantization_error = torch.norm(net.linear.weight - quantized_weights)
print(f"\n量化误差（权重）: {quantization_error.item()}")

# 创建输入数据
input_data = torch.randn(1, input_dim)

# 前向传播计算原始输出
original_output = net(input_data)

# 重新量化权重并计算量化后的输出
with torch.no_grad():
    for name, param in net.named_parameters():
        if "weight" in name:
            param.copy_(quantized_weights)

# 前向传播计算量化后的输出
quantized_output = net(input_data)

# 计算输出误差
output_error = torch.norm(original_output - quantized_output)
print(f"量化后的输出误差: {output_error.item()}")
import torch

# 创建一个张量，可以是任意形状，这里以一维张量为例
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 应用sigmoid函数
y = torch.sigmoid(x)

print("原始张量:", x)
print("Sigmoid 后的张量:", y)
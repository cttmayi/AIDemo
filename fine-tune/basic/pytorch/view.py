import torch

# 创建一个形状为 (3, 4) 的张量
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

print("原始张量:", tensor)

# 使用 unsqueeze 在第1维上增加一个维度，新的张量形状为 (3, 1, 4)
tensor_view = tensor.view(-1)

print("使用 view 后的张量:", tensor_view)
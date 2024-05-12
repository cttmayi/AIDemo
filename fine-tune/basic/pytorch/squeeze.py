import torch

# 创建一个形状为 (1, 3, 1, 4) 的张量
tensor = torch.tensor([[[1, 2, 3, 4]],
                      [[5, 6, 7, 8]],
                      [[9, 10, 11, 12]]])

print("原始张量:", tensor)

# 使用 squeeze 移除所有大小为1的维度，新的张量形状为 (3, 4)
tensor_squeezed = tensor.squeeze()

print("使用 squeeze 后的张量:", tensor_squeezed)
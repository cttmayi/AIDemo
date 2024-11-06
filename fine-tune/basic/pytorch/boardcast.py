# 广播机制

import torch

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(2, 2)
print('a', a)
print('b', b)
# (3, 1) + (1, 2) = (3, 2) + (3, 2) = (3, 2)
print('a + b', a + b)  # 广播机制
# a + b tensor([[0, 1],
#         [1, 2],
#         [2, 3]])


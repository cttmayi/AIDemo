# 引入库
import torch

# 生成张量
A = torch.arange(25).reshape(5, -1)
print(A)

print('A[0]', A[0])
print('A[0, 1]', A[0, 1])
print('A[0, 0:2]', A[0, 0:2])
print('A[0, 1:3]', A[0, 1:3])

print(A.sum(axis=1))
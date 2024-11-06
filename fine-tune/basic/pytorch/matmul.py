import torch

a = torch.ones(3, 4)
b = torch.ones(4, 4, 2)

c = torch.matmul(a,b)

print('a', a, a.shape)
print('b', b, b.shape)
print('c', c, c.shape)

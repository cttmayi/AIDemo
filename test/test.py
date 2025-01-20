import torch


a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = torch.Tensor([1, 2, 4, 4, 5, 6, 7, 8, 9, 9])

print(torch.eq(a, b).all().item())
print(torch.where(a != b)[0].tolist())
print((a != b).all().item())
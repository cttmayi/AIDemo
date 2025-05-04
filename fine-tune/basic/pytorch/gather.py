import torch
import torch.nn.functional as F

# 创建一个原始张量
input_tensor = torch.tensor([[[1.0, 2.0, 3.0],
                              [0.4, 5.0, 6.0]]])



# 创建一个索引张量，它将决定从input_tensor中提取哪些元素
index_tensor = torch.tensor([[0, 2, 0],
                              [1, 1, 2]])

# 使用gather函数根据index_tensor中的索引提取input_tensor中的元素
# 假设我们想沿着第一个维度（dim=0）进行操作
output = torch.gather(input_tensor.squeeze(0), 1, index_tensor)

print("原始张量:\n", input_tensor)
print("索引张量:\n", index_tensor)
print("提取后的张量:\n", output)



# 提取后的张量:
#  tensor([[1, 3, 1],
#         [5, 5, 6]])

# input_tensor = F.softmax(input_tensor, dim=-1)

index_tensor = torch.tensor([[0, 1]])
index_tensor = index_tensor.transpose(0, 1)#  .reshape(3, 1)

output = torch.gather(input_tensor.squeeze(0), 1, index_tensor)

print('----------------------')
print("原始张量:\n", input_tensor)
print("索引张量:\n", index_tensor)
print("提取后的张量:\n", output)
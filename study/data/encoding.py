import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


xi = torch.tensor([2, 5, 8])
xf = torch.tensor([0.1, 0.5, 0.9])

######################################################
# 热独编码(One-Hot)
# 适用场景：适用于类别之间无顺序关系的分类变量。
# 优点：信息无损，能完整保留每个选项的信息；避免误导模型，不会引入错误的顺序关系。
# 缺点：维度灾难，输入维度会急剧增加；稀疏性，大部分元素为0，可能影响训练效率。
one_hot = F.one_hot(xi, num_classes=10)
print("热独编码:")
print(one_hot)

###################################################
# 嵌入编码（Embedding）
# 适用场景：适用于类别数量较多的分类变量。
# 优点：可以将高维的类别信息映射到低维的向量空间中，减少维度；能够学习到类别之间的相似性和关系。
# 缺点：需要额外的训练来学习嵌入向量，增加了模型的复杂度。
embed_dim = 6
embedding = nn.Embedding(9, embed_dim)  # 有3个选项
x_embed = embedding(xi)
print("嵌入编码:")
print(x_embed)


###################################################
# 标签编码
# 适用场景：适用于类别数量较少且类别之间有顺序关系的分类变量。
# 优点：能够保留类别之间的顺序关系；实现简单，不需要额外的训练。
# 缺点：容易受到异常值的影响；可能引入误导性的顺序关系。
x_encoded = xi - xi.min()
print("标签编码")
print(x_encoded)


###################################################
# 离散编码
# 适用场景：适用于需要将连续变量离散化的场景。
# 优点：能够将连续变量离散化，减少计算复杂度；能够捕捉数据的离散特征。
# 缺点：可能丢失连续变量的部分信息；需要选择合适的离散化方法。
x_discrete = xf.round()
print("离散编码:")
print(x_discrete)


###################################################
# 二机制编码(Binary Encoding)
# 适用场景：适用于需要将连续变量离散化的场景。
# 优点：能够将连续变量离散化，减少计算复杂度；能够捕捉数据的离散特征。
# 缺点：可能丢失连续变量的部分信息；需要选择合适的离散化方法。
def int_to_binary(num, num_bits):
    binary_array = np.zeros(num_bits, dtype=int)
    for i in range(num_bits):
        binary_array[i] = num % 2
        num //= 2
    binary_array = binary_array[::-1]
    return binary_array

num = 10
num_bits = 8
binary_array = int_to_binary(num, num_bits)
print("二进制编码:")
print(binary_array)  # 输出：[0 0 0 0 1 0 1 0]

###################################################
# 时延编码
# 适用场景：适用于需要利用时间属性来表示数据大小的场景。
# 优点：通过脉冲发放时间来表示数据大小，能够捕捉数据的时间特征。
# 缺点：计算脉冲发放时间的公式较为复杂，实现难度相对较高。

T = 100  # 总时间窗
a = 1.0  # 缩放因子
T_first = T - 1 - torch.log(a * xf + 1)
print("延时编码:")
print(T_first)
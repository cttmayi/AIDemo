import numpy as np

# 3 x 5 , 3维特征，5组数据
data = np.array([
    [ -1, -1,  0,  2,  1],
    [  2,  0,  0, -1, -1],
    [  2,  0,  1,  1,  0]], 
    dtype=np.float64)

n, m = data.shape
k = 2
data = data - data.mean(axis=1, keepdims=True)
# 协方差矩阵
C = np.dot(data, data.T) / m
# u的每一列是特征向量
u, d, v = np.linalg.svd(C)
new_data  = np.dot(u[:,:k].T, data)

print(new_data)

#  输出
#[[ 2.50856792  0.59089386  0.19057608 -1.84517782 -1.44486004]
# [-0.76054613  1.31546989 -0.02787302 -0.93519683  0.40814608]]

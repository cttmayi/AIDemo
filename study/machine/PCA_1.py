import numpy as np
from sklearn.decomposition import PCA

# 3 x 5 , 3维特征，5组数据
data = np.array([
    [-1, -1,  0,  2,  1],
    [ 2,  0,  0, -1, -1],
    [ 2,  0,  1,  1,  0]],
    dtype = np.float64)

k = 2

pca = PCA(n_components=k)
# 注意转置
new_data = pca.fit_transform(data.T)
# 降维后的数据各个维度特征 所占信息比例
ratios = pca.explained_variance_ratio_

print(new_data)
print(ratios)

import torch

# 假设self.config.score_clip是一个配置类中的属性，这里我们直接设置一个值
score_clip = 1.0

# 创建一个随机张量scores，这里只是示例，所以使用torch.randn生成随机数
scores = torch.randn(3, 5)  # 生成一个3行5列的张量

# 转换为浮点数类型并裁剪，确保scores中的值在[-score_clip, score_clip]范围内
scores_clipped = torch.clip(scores.float(), -score_clip, score_clip)

# 假设我们想要将裁剪后的张量转换为double类型
scores_dtype = torch.float64
scores_clipped = scores_clipped.to(dtype=scores_dtype)

# 打印裁剪并转换后的张量
print(scores_clipped)
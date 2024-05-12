import torch
import torch.nn.functional as F

# 假设我们有两个概率分布，这里我们使用两个简单的张量来表示
# 这些张量应该表示对数概率，因为log_target=True
ref_logprob = torch.tensor([0.1, 0.2, 0.7])  # 参考概率分布的对数概率
logprob = torch.tensor([0.3, 0.3, 0.4])      # 模型输出的概率分布的对数概率

# 使用F.kl_div计算KL散度
kl_divergence = F.kl_div(ref_logprob, logprob, log_target=False, reduction="none")

# 累加最后一个维度的KL散度，得到一个总的损失值
kl_loss = kl_divergence.sum(-1)

# 打印KL散度和总的损失值
print("KL Divergence:", kl_divergence)
print("Total KL Loss:", kl_loss)


# 输出结果(if log_target=True)
# KL Divergence: tensor([ 0.2700,  0.1350, -0.4475])
# Total KL Loss: tensor(-0.0426)

# 输出结果(if log_target=False)
# KL Divergence: tensor([-0.3912, -0.4212, -0.6465])
# Total KL Loss: tensor(-1.4589)
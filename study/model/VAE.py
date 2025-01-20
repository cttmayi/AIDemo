import torch
import torch.nn as nn
import torch.nn.functional as F

# 变分自编码器（Variational Autoencoder, VAE）

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar = z.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# 示例
input_dim = 784  # 28x28图像展平后的维度
hidden_dim = 128
latent_dim = 2  # 潜在变量的维度

model = VAE(input_dim, hidden_dim, latent_dim)
x = torch.randn(1, input_dim)  # 生成一个随机输入
reconstructed, mu, logvar = model(x)

print("重构的图像：", reconstructed)
print("潜在变量的均值：", mu)
print("潜在变量的方差：", logvar)
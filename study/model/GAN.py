import torch
import torch.nn as nn

# 生成对抗网络（Generative Adversarial Network, GAN）

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 示例
z_dim = 100  # 随机噪声的维度
img_dim = 784  # 28x28图像展平后的维度

generator = Generator(z_dim, img_dim)
discriminator = Discriminator(img_dim)

z = torch.randn(1, z_dim)  # 生成一个随机噪声
generated_img = generator(z)
validity = discriminator(generated_img)

print("生成的图像：", generated_img)
print("判别器输出：", validity)
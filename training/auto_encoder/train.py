import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import os

from utils.model import create_model, save_model
from utils.data import create_dataset


def train_or_eval(model, name, data_loader, loss_fn, device, optimizer=None):
    train_loss = 0
    train_nsample = 0
    t = tqdm(data_loader, desc = f'[{name}]epoch:{epoch}')
    for imgs, lbls in t: #imgs:(bs,28,28)
        bs = imgs.shape[0]
        imgs = imgs.to(device).view(bs,input_size) #imgs:(bs,28*28)
        re_imgs = model(imgs)
        loss = loss_fn(re_imgs, imgs) # 重构与原始数据的差距
        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #计算平均损失，设置进度条
        train_loss += loss.item()
        train_nsample += bs
        t.set_postfix({'loss':train_loss/train_nsample})
    return train_loss/train_nsample


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    #损失函数
    loss_BCE = torch.nn.BCELoss(reduction = 'sum')    # 交叉熵，衡量各个像素原始数据与重构数据的误差
    loss_MSE = torch.nn.MSELoss(reduction = 'sum')    # 均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差

    #模型参数
    latent_size = 256 #压缩后的特征维度
    hidden_size = 1024 #encoder和decoder中间层的维度
    input_size = output_size = 28 * 28 #原始图片和生成图片的维度

    #训练参数
    epochs = 100 #训练时期
    batch_size = 32 #每步训练样本数
    learning_rate = 1e-5 #学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

    #确定模型，导入已训练模型（如有）
    model_name = 'ae.pth'

    model = create_model(input_size, output_size, latent_size, hidden_size, device, model_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #准备mnist数据集
    data_path = './.data'
    train_loader, test_loader = create_dataset('mnist', data_path, batch_size)

    #训练及测试
    loss_history = {'train':[],'eval':[]}
    for epoch in range(epochs):   
        #训练
        model.train()
        loss = train_or_eval(model, 'train', train_loader, loss_BCE, device, optimizer)
        loss_history['train'].append(loss)
        save_model(model, model_name)

        #测试
        model.eval()
        loss = train_or_eval(model, 'eval', test_loader, loss_MSE, device) 
        loss_history['eval'].append(loss)

        #展示效果
        imgs = test_loader.dataset[0][0].to(device).view(1,-1)
        re_imgs = model(imgs)
        concat = torch.cat((imgs[0].view(28, 28),
                re_imgs[0].view( 28, 28)), 1)
        plt.matshow(concat.cpu().detach().numpy())

        

    #显示每个epoch的loss变化
    plt.figure(figsize=(10,5))
    plt.plot(range(epoch+1),loss_history['train'])
    plt.plot(range(epoch+1),loss_history['eval'])
    plt.show()


    #对数据集
    dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    #取一组数据
    raw = dataset[0][0].view(1,-1) #raw: bs,28,28->bs,28*28
    #用encoder压缩数据
    feat = model.encoder(raw)
    #展示数据及维度
    print(raw.shape,'->',feat.shape)
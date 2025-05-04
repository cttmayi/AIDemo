import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import utils.cfg as cfg




class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, dtype=cfg.DTYPE)
        # self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, latent_size, dtype=cfg.DTYPE)

    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear1(x)) #-> bs,hidden_size
        # x = self.dropout(x)
        x = self.linear2(x) #-> bs,latent_size
        return x

class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size, dtype=cfg.DTYPE)
        self.linear2 = torch.nn.Linear(hidden_size, output_size, dtype=cfg.DTYPE)   

    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        # x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        x = self.linear2(x) #->bs,output_size
        x = F.sigmoid(x)
        return x

class AE(torch.nn.Module):
    #将编码器解码器组合，数据先后通过编码器、解码器处理
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)
    def forward(self, x): #x: bs,input_size
        feat = self.encoder(x) #feat: bs,latent_size
        re_x = self.decoder(feat) #re_x: bs, output_size
        return re_x
    
    def encode(self, x):
        return self.encoder(x)
    

def create_model(input_size, output_size, latent_size, hidden_size, model_name=None):
    model = AE(input_size, output_size, latent_size, hidden_size)
    try:
        if model_name is not None:
            model.load_state_dict(torch.load(model_name))
            print('[INFO] Load Model complete')
    except:
        pass
    return model

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name)
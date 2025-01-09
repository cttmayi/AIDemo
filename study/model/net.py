import torch
import torch.nn.functional as F
import torch.nn as nn


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return F.log_softmax(x, dim=1)
    

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.l = nn.Linear(input_size, output_size)
        # self.h = nn.ReLU()
        self.h = nn.Softmax()
    
    def forward(self, x):
        x = self.l(x)
        # x = self.h(x)
        return x
    



if __name__ == '__main__':
    net = Net(10, 10)
    print(net)
    x = torch.randn(1, 10)
    print(net(x))
    
import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d, relu

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.fcl1 = nn.Linear(784, 500)
        self.fcl2 = nn.Linear(500, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x, 2)
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = relu(self.fcl1(x))
        return self.fcl2(x)

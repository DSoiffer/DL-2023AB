import torch.nn as nn
import torch as torch

class Conv2(nn.Module):
    def __init__(self, activation):
        super(Conv2, self).__init__()
        self.flatten_img_dim = int(32 / 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.activation = activation
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.fc1 = nn.Linear(64 * self.flatten_img_dim ** 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        h = self.activation(self.conv1_1(x))
        h = self.pool(self.activation(self.conv1_2(h)))

        h = h.view(-1, 64 * self.flatten_img_dim ** 2)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h
    
class Basic(nn.Module):
    def __init__(self, activation):
        super(Basic, self).__init__()
        self.activation = activation
        self.input_size = (32 ** 2) * 3
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        h = x.view(x.shape[0], -1)
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h
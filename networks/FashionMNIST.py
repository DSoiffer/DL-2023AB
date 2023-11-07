import torch.nn as nn

class Basic(nn.Module): 
    def __init__(self, activation): 
        super(Basic, self).__init__() 
        self.fc1 = nn.Linear(784, 128) 
        self.activation = activation
        self.fc2 = nn.Linear(128, 10) 
      
    def forward(self, x): 
        x = x.view(-1, 784) 
        x = self.activation(self.fc1(x)) 
        x = self.fc2(x) 
        return x
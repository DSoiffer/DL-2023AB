import torch.nn as nn
import torch.nn.functional as F

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

class Standard(nn.Module):
    def __init__(self, activation):
        super(Standard, self).__init__()
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120,10)
        self.activation = activation
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        #not using dropout on output layer
        x = F.log_softmax(self.fc3(x), dim=1)
        return x   

class Deep(nn.Module):
    def __init__(self, activation):
        super(Deep, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3), padding=1)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 20, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation

    def forward(self, x):
        x = self.dropout_2d(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.dropout_2d(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(-1, 7 * 7 * 20)  # flatten / reshape
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
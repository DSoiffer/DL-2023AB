import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import data_loader.FashionMNIST as FashionMNIST_loader

def abs_activation(x):
    return torch.abs(x)


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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        # Reset gradients to 0, since this is not done automatically
        optimizer.zero_grad()

        if batch % (150*64/batch_size) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    

# This function adapted from PyTorch's "Getting Started" examples
def test_loop(dataloader, model, loss_fn, showAcc=True):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if showAcc:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    if showAcc:
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


training_data, validation_data, test_data = FashionMNIST_loader.load(True)
learning_rate = 1e-2
batch_size = 16
epochs = 10
alpha = 0.001

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


absModel = Basic(abs_activation)
reluModel = Basic(nn.ReLU())

loss_fn = nn.CrossEntropyLoss()
optimizerAbs = torch.optim.SGD(absModel.parameters(), lr=learning_rate, weight_decay=alpha)
optimizerReLU = torch.optim.SGD(reluModel.parameters(), lr=learning_rate, weight_decay=alpha)


def runModel(model, optimizer):
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        traj_params = train_loop(train_dataloader, model, loss_fn, optimizer)
        # Print the validation accuracy
        test_loop(valid_dataloader, model, loss_fn)


   # test_loop(test_dataloader, model, loss_fn)  # ONLY USE ONCE ALL HYPERPARAMS OPTIMIZED!
   
runModel(absModel, optimizerAbs)
runModel(reluModel, optimizerReLU)

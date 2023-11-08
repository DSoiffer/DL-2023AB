import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import data_loader.FashionMNIST as FashionMNIST_loader
import networks.FashionMNIST as FashionMNIST_networks
from setup import abs_activation, runModel

training_data, validation_data, test_data = FashionMNIST_loader.load(True)
learning_rate = 1e-2
batch_size = 16
epochs = 1
alpha = 0.001

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


absModel = FashionMNIST_networks.Basic(abs_activation)
reluModel = FashionMNIST_networks.Basic(nn.ReLU())

loss_fn = nn.CrossEntropyLoss()
optimizerAbs = torch.optim.SGD(absModel.parameters(), lr=learning_rate, weight_decay=alpha)
optimizerReLU = torch.optim.SGD(reluModel.parameters(), lr=learning_rate, weight_decay=alpha)


epochsAbs, lossAbs, accAbs, timeAbs = runModel(absModel, train_dataloader, val_dataloader, test_dataloader, optimizerAbs, loss_fn, True, (epochs, batch_size, 3, .1))
epochsRelu, lossRelu, accRelu, timeRelu = runModel(reluModel, train_dataloader, val_dataloader, test_dataloader, optimizerReLU, loss_fn, True, (epochs, batch_size, 3, .1))

print("Abs model results: epochs:",epochsAbs,"loss:", lossAbs, "accuracy:", accAbs*100, "time:", timeAbs)
print("Relu model results: epochs:",epochsRelu,"loss:", lossRelu, "accuracy:", accRelu*100, "time", timeRelu)
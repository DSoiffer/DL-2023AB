import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import data_loader.FashionMNIST as FashionMNIST_loader
import data_loader.CIFAR10 as CIFAR10_loader
import networks.FashionMNIST as FashionMNIST_networks
import networks.CIFAR10 as CIFAR10_networks
from setup import abs_activation, runModel, hyper_tuning, test_loop

#training_data, validation_data, test_data = FashionMNIST_loader.load(True)
training_data, validation_data, test_data = CIFAR10_loader.load(True)
learning_rate = 1e-2
batch_size = 16
epochs = 1
alpha = 0.001

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# absModel = FashionMNIST_networks.Basic(abs_activation)
# reluModel = FashionMNIST_networks.Basic(nn.ReLU())
absModel = CIFAR10_networks.Standard(abs_activation)
reluModel = CIFAR10_networks.Standard(nn.ReLU())

loss_fn = nn.CrossEntropyLoss()
# optimizerAbs = torch.optim.SGD(absModel.parameters(), lr=learning_rate, weight_decay=alpha)
# optimizerReLU = torch.optim.SGD(reluModel.parameters(), lr=learning_rate, weight_decay=alpha)


# epochsAbs, lossAbs, accAbs, timeAbs = runModel(absModel, train_dataloader, val_dataloader, test_dataloader, optimizerAbs, loss_fn, True, (epochs, batch_size, 3, .1))
# epochsRelu, lossRelu, accRelu, timeRelu = runModel(reluModel, train_dataloader, val_dataloader, test_dataloader, optimizerReLU, loss_fn, True, (epochs, batch_size, 3, .1))

# print("Abs model results: epochs:",epochsAbs,"loss:", lossAbs, "accuracy:", accAbs*100, "time:", timeAbs)
# print("Relu model results: epochs:",epochsRelu,"loss:", lossRelu, "accuracy:", accRelu*100, "time", timeRelu)

batch_sizes = [16]
learning_rates = [1e-4, 1e-2]
alphas = [0.001, .01]
epochs = [1, 2]

if torch.cuda.is_available(): 
  dev = "cuda:0" 
else: 
  dev = "cpu" 
device = torch.device(dev) 

resAbs = hyper_tuning(absModel, train_dataloader, val_dataloader, loss_fn, (batch_sizes, learning_rates, alphas, epochs))
print("Best abs hyperparams:", resAbs[2])
lossAbs, accAbs = test_loop(test_dataloader, absModel, loss_fn)
print("Abs model results: epochs:",resAbs[2][3],"loss:", lossAbs, "accuracy:", accAbs*100, "time:", resAbs[4])

resRelu = hyper_tuning(reluModel, train_dataloader, val_dataloader, loss_fn, (batch_sizes, learning_rates, alphas, epochs))
print("Best relu hyperparams:", resRelu[2])
lossRelu, accRelu = test_loop(test_dataloader, reluModel, loss_fn)
print("Relu model results: epochs:",resRelu[2][3],"loss:", lossRelu, "accuracy:", accRelu*100, "time:", resRelu[4])

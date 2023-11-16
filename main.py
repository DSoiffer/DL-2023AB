import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import json

import data_loader.FashionMNIST as FashionMNIST_loader
import data_loader.CIFAR10 as CIFAR10_loader
import networks.FashionMNIST as FashionMNIST_networks
import networks.CIFAR10 as CIFAR10_networks
from setup import abs_activation, runModel, hyper_tuning, test_loop

# the options we have so far 
# TODO move to another file?
options = {
  "FashionMNIST": {
    'loader': FashionMNIST_loader,
    'models': {
      'basic':  FashionMNIST_networks.Basic
    },
    'showAcc': True
  },
  "CIFAR10": {
    'loader': CIFAR10_loader,
    'models': {
      'standard': CIFAR10_networks.Standard
    },
    'showAcc': True
  }
}
# TODO showAcc feature isn't used much, kinda assumed

# define what you're choosing here
data = 'CIFAR10'
model = 'standard'
out_file = 'res/CIFAR10_standard.json'

training_data, validation_data, test_data = options[data]['loader'].load(True)

absModel = options[data]['models'][model](abs_activation)
reluModel = options[data]['models'][model](nn.ReLU())

loss_fn = nn.CrossEntropyLoss() # TODO do we need to have the ability to change this?

batch_sizes = [16]
learning_rates = [1e-4, 1e-2]
alphas = [0.001, .01]
epochs = [1, 2]

if torch.cuda.is_available(): 
  dev = "cuda:0" 
else: 
  dev = "cpu" 
device = torch.device(dev) 

resAbs, hyperAbs = hyper_tuning(absModel, training_data, validation_data, loss_fn, (batch_sizes, learning_rates, alphas, epochs))
test_dataloader = DataLoader(test_data, batch_size=hyperAbs['batch_size']) # account for batch_size as a hyperparam
lossAbs, accAbs = test_loop(test_dataloader, absModel, loss_fn)
print("Abs model best results: hyperparams:",hyperAbs, "model results:", resAbs, "loss:", lossAbs, "accuracy:", accAbs*100)

resRelu, hyperRelu  = hyper_tuning(reluModel, training_data, validation_data, loss_fn, (batch_sizes, learning_rates, alphas, epochs))
test_dataloader = DataLoader(test_data, batch_size=hyperRelu['batch_size'])
lossRelu, accRelu = test_loop(test_dataloader, reluModel, loss_fn)
print("Relu model hypertuned results: hyperparams:",hyperRelu, "model results:", resRelu, "loss:", lossRelu, "accuracy:", accRelu*100)

# now doing one model with the best hyperparams for the other one
# running relu using best of abs
train_dataloader = DataLoader(training_data, batch_size=hyperAbs['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyperAbs['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyperAbs['batch_size'])
optimizer = torch.optim.SGD(reluModel.parameters(), lr=hyperAbs['learning_rate'], weight_decay=hyperAbs['alpha'], momentum=.5) #TODO momentum hyperparameter?
resRelu2 = runModel(reluModel, train_dataloader, val_dataloader, optimizer, loss_fn, True, (hyperAbs['epochs'], hyperAbs['batch_size'], 3, .1))
lossRelu2, accRelu2 = test_loop(test_dataloader, reluModel, loss_fn)
print("Relu model based off of abs results: hyperparams:",hyperAbs, "model results:", resRelu2, "loss:", lossRelu2, "accuracy:", accRelu2*100)
# running abs using best of relu
train_dataloader = DataLoader(training_data, batch_size=hyperRelu['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyperRelu['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyperRelu['batch_size'])
optimizer = torch.optim.SGD(absModel.parameters(), lr=hyperRelu['learning_rate'], weight_decay=hyperRelu['alpha'], momentum=.5) #TODO momentum hyperparameter?
resAbs2 = runModel(absModel, train_dataloader, val_dataloader, optimizer, loss_fn, True, (hyperRelu['epochs'], hyperRelu['batch_size'], 3, .1))
lossAbs2, accAbs2 = test_loop(test_dataloader, reluModel, loss_fn)
print("Abs model based off of relu results: hyperparams:",hyperRelu, "model results:", resAbs2,  "loss:", lossAbs2, "accuracy:", accAbs2*100)


fileRes = {
  'abs': {
    'hypertuned': {
      'hyper': hyperAbs,
      'model results': resAbs,
      'loss': lossAbs,
      'accuracy': accAbs * 100
    },
    'otherHypertuned': {
      'hyper': hyperRelu,
      'model results': resAbs2,
      'loss': lossAbs2,
      'accuracy': accAbs2 * 100
    }
  },
  'relu': {
    'hypertuned': {
      'hyper': hyperRelu,
      'model results': resRelu,
      'loss': lossRelu,
      'accuracy': accRelu * 100
    },
    'otherHypertuned': {
      'hyper': hyperAbs,
      'model results': resRelu2,
      'loss': lossRelu2,
      'accuracy': accRelu2 * 100
    }
  }
}
with open(out_file, 'w') as f:
  json.dump(fileRes, f)


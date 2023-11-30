import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pandas as pd

import data_loader.FashionMNIST as FashionMNIST_loader
import data_loader.CIFAR10 as CIFAR10_loader
import networks.FashionMNIST as FashionMNIST_networks
import networks.CIFAR10 as CIFAR10_networks
from setup import abs_activation, runModel, hyper_tuning, test_loop, plot, PyTorchClassifier

# the options we have so far 
# TODO move to another file?
options = {
  "FashionMNIST": {
    'loader': FashionMNIST_loader,
    'models': {
      'basic':  FashionMNIST_networks.Basic,
      'standard': FashionMNIST_networks.Standard,
      'deep': FashionMNIST_networks.Deep
    },
    'show_acc': True
  },
  "CIFAR10": {
    'loader': CIFAR10_loader,
    'models': {
      'standard': CIFAR10_networks.Standard
    },
    'show_acc': True
  }
}

# define what you're choosing here
data = 'FashionMNIST'
model = 'deep'
out_file = 'res/FashionMNIST_deep.json'

param_grid = {
    'batch_size': [64, 128],
    'learning_rate': [0.001, 0.01],
    'alpha': [0.0001, 0.001],
    'epochs': [1, 50],
    'momentum': [.5],
    'patience': [3],
    'min_delta': [.1]
}


# training_data, validation_data, test_data = options[data]['loader'].load(True)
train_val_data, test_data = options[data]['loader'].load(False)
training_data, validation_data = torch.utils.data.random_split(train_val_data, [.85, .15])

# setting up validation set, getting indices for it for hypertuning
train_val_x = []
train_val_y = []
for i in range(len(train_val_data)):
  train_val_x.append(train_val_data[i][0])
  train_val_y.append(train_val_data[i][1])
train_val_x = torch.stack(train_val_x)

val_indices = [-1]*len(train_val_data)
for v in validation_data.indices:
  val_indices[v] = 0
# splitting on validation indices
ps = PredefinedSplit(test_fold=val_indices)

absModel = options[data]['models'][model](abs_activation)
reluModel = options[data]['models'][model](nn.ReLU())
absModel2 = options[data]['models'][model](abs_activation)
reluModel2 = options[data]['models'][model](nn.ReLU())
show_acc = options[data]['show_acc']

loss_fn = nn.CrossEntropyLoss() # TODO do we need to have the ability to change this? do this every time

# potential speed up
if torch.cuda.is_available(): 
  dev = "cuda:0" 
else: 
  dev = "cpu" 
device = torch.device(dev) 

abs_classifier = PyTorchClassifier(absModel, 0,0,0,0,0,0,0)
abs_grid_search = GridSearchCV(abs_classifier, param_grid, cv=ps, verbose=2, scoring="accuracy", n_jobs=-1)
abs_grid_search.fit(train_val_x, train_val_y)
hyper_abs = abs_grid_search.best_params_
print("Abs best hyperparams: ", hyper_abs)

relu_classifier = PyTorchClassifier(reluModel, 0,0,0,0,0,0,0)
#cv=ps
relu_grid_search = GridSearchCV(relu_classifier, param_grid, cv=ps, verbose=2, scoring="accuracy", n_jobs=-1)
relu_grid_search.fit(train_val_x, train_val_y)
hyper_relu = relu_grid_search.best_params_
print("Relu best hyperparams: ", hyper_relu)

# running abs using best of abs
loss_fn = nn.CrossEntropyLoss()
train_dataloader = DataLoader(training_data, batch_size=hyper_abs['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyper_abs['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyper_abs['batch_size'])
optimizer = torch.optim.SGD(absModel.parameters(), lr=hyper_abs['learning_rate'], weight_decay=hyper_abs['alpha'], momentum=hyper_abs['momentum'])
resAbs = runModel(absModel, train_dataloader, val_dataloader, optimizer, loss_fn, (hyper_abs['epochs'], hyper_abs['batch_size'], 3, .1), show_acc)
if show_acc:
  lossAbs, accAbs = test_loop(test_dataloader, absModel, loss_fn, show_acc)
  print("Abs model based off of abs hyperparams: hyperparams:",hyper_abs, "\nmodel results:", resAbs, "\nloss:", lossAbs, "accuracy:", accAbs*100)
else:
  lossAbs = test_loop(test_dataloader, absModel, loss_fn, show_acc)
  print("Abs model based off of abs hyperparams: hyperparams:",hyper_abs, "\nmodel results:", resAbs, "\nloss:", lossAbs)

# # running abs using best of relu
loss_fn = nn.CrossEntropyLoss()
train_dataloader = DataLoader(training_data, batch_size=hyper_relu['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyper_relu['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyper_relu['batch_size'])
optimizer = torch.optim.SGD(absModel2.parameters(), lr=hyper_relu['learning_rate'], weight_decay=hyper_relu['alpha'], momentum=hyper_relu['momentum'])
resAbs2 = runModel(absModel2, train_dataloader, val_dataloader, optimizer, loss_fn, (hyper_relu['epochs'], hyper_relu['batch_size'], 3, .1), show_acc)
if show_acc:
  lossAbs2, accAbs2 = test_loop(test_dataloader, absModel, loss_fn, show_acc)
  print("Abs model based off of relu hyperparams: hyperparams:",hyper_relu, "\nmodel results:", resAbs2, "\nloss:", lossAbs2, "accuracy:", accAbs2*100)
else:
  lossAb2s = test_loop(test_dataloader, absModel, loss_fn, show_acc)
  print("Abs model based off of relu hyperparams: hyperparams:",hyper_relu, "\nmodel results:", resAbs2, "\nloss:", lossAbs2)


# running relu using best of relu
loss_fn = nn.CrossEntropyLoss()
train_dataloader = DataLoader(training_data, batch_size=hyper_relu['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyper_relu['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyper_relu['batch_size'])
optimizer = torch.optim.SGD(reluModel.parameters(), lr=hyper_relu['learning_rate'], weight_decay=hyper_relu['alpha'], momentum=hyper_relu['momentum'])
resRelu = runModel(reluModel, train_dataloader, val_dataloader, optimizer, loss_fn, (hyper_relu['epochs'], hyper_relu['batch_size'], 3, .1), show_acc)
if show_acc:
  lossRelu, accRelu = test_loop(test_dataloader, reluModel, loss_fn, show_acc)
  print("Relu model based off of relu hyperparams: hyperparams:",hyper_relu, "\nmodel results:", resRelu, "\nloss:", lossRelu, "accuracy:", accRelu*100)
else:
  lossRelu = test_loop(test_dataloader, reluModel, loss_fn, show_acc)
  print("Relu model based off of relu hyperparams: hyperparams:",hyper_relu, "\nmodel results:", resRelu, "\nloss:", lossRelu)

# running relu using best of abs
loss_fn = nn.CrossEntropyLoss()
train_dataloader = DataLoader(training_data, batch_size=hyper_abs['batch_size']) 
val_dataloader = DataLoader(validation_data, batch_size=hyper_abs['batch_size'])
test_dataloader = DataLoader(test_data, batch_size=hyper_abs['batch_size'])
optimizer = torch.optim.SGD(reluModel2.parameters(), lr=hyper_abs['learning_rate'], weight_decay=hyper_abs['alpha'], momentum=hyper_abs['momentum'])
resRelu2 = runModel(reluModel2, train_dataloader, val_dataloader, optimizer, loss_fn, (hyper_abs['epochs'], hyper_abs['batch_size'], 3, .1), show_acc)
if show_acc:
  lossRelu2, accRelu2 = test_loop(test_dataloader, reluModel, loss_fn, show_acc)
  print("Relu model based off of abs hyperparams: hyperparams:",hyper_abs, "\nmodel results:", resRelu2, "\nloss:", lossRelu2, "accuracy:", accRelu2*100)
else:
  lossRelu2 = test_loop(test_dataloader, reluModel, loss_fn, show_acc)
  print("Relu model based off of abs hyperparams: hyperparams:",hyper_abs, "\nmodel results:", resRelu2, "\nloss:", lossRelu2)


fileRes = {
  'abs': {
    'hypertuned': {
      'hyper': hyper_abs,
      'model results': resAbs,
      'loss': lossAbs,
      'accuracy': None if not show_acc else accAbs * 100
    },
    'otherHypertuned': {
      'hyper': hyper_relu,
      'model results': resAbs2,
      'loss': lossAbs2,
      'accuracy': None if not show_acc else accAbs2 * 100
    }
  },
  'relu': {
    'hypertuned': {
      'hyper': hyper_relu,
      'model results': resRelu,
      'loss': lossRelu,
      'accuracy': None if not show_acc else accRelu * 100
    },
    'otherHypertuned': {
      'hyper': hyper_abs,
      'model results': resRelu2,
      'loss': lossRelu2,
      'accuracy': None if not show_acc else accRelu2 * 100
    }
  }
}
with open(out_file, 'w') as f:
  json.dump(fileRes, f)


lossDict = {
  'abs hyp trn' : resAbs['train_losses'],
  'abs hyp val' : resAbs['val_losses'],
  'relu hyp trn' : resRelu['train_losses'],
  'relu hyp val' : resRelu['val_losses'],
  'abs noth trn' : resAbs2['train_losses'],
  'abs noth val' : resAbs2['val_losses'],
  'relu noth trn' : resRelu2['train_losses'],
  'relu noth val' : resRelu2['val_losses'],
}

plot(lossDict, data + " " + model)

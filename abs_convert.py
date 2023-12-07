import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
import os
import random
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
import shutil
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
print(device)

def get_data(augmentation=0):
  # Data augmentation transformations. Not for Testing!
  if augmentation:
    transform_train = transforms.Compose([
      transforms.Resize(128),
      transforms.RandomCrop(128, padding=8, padding_mode='edge'), # Take 128x128 crops from 136x136 padded images
      transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
      transforms.ToTensor(),
    ])
  else:
    transform_train = transforms.ToTensor()

  transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
  ])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

  # generator = torch.Generator().manual_seed(42)
  # trainset, valset = random_split(trainset, [.8, .2], generator=generator)
  # trainset.transform = transform_train
  # valset.transform = transform_test
  # print(trainset, valset)
  # # print(trainset.transform)
  # print(valset.transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                            num_workers=0)
  # valloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
  #                                           num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                      transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                          num_workers=0)
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  return {'train': trainloader, 'test': testloader, 'classes': classes}

def train(net, train_dataloader, val_dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005,
          verbose=1, gamma=0.9, print_every=10, state=None, schedule={}, checkpoint_path=None, output_file=None):
  net.to(device)
  net.train()
  losses = []
  val_losses = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  scheduler = ExponentialLR(optimizer, gamma=gamma)

  # Load previous training state
  if state:
      net.load_state_dict(state['net'])
      optimizer.load_state_dict(state['optimizer'])
      start_epoch = state['epoch']
      losses = state['losses']
      val_losses = state['test_losses']

  # Fast forward lr schedule through already trained epochs
  for epoch in range(start_epoch):
    if epoch in schedule:
      print ("Learning rate: %f"% schedule[epoch])
      for g in optimizer.param_groups:
        g['lr'] = schedule[epoch]

  for epoch in range(start_epoch, epochs):
    sum_loss = 0.0
    amt_losses = 0
    print_sum_loss = 0.0
    print_amt_losses = 0

    # Update learning rate when scheduled
    if epoch in schedule:
      print ("Learning rate: %f"% schedule[epoch])
      for g in optimizer.param_groups:
        g['lr'] = schedule[epoch]
    for i, batch in enumerate(train_dataloader, 0):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        #print(i)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # autograd magic, computes all the partial derivatives
        optimizer.step() # takes a step in gradient direction

        sum_loss += loss.item()
        amt_losses += 1
        print_sum_loss += loss.item()
        print_amt_losses += 1
        

        if i % print_every == print_every-1:    # print every 10 mini-batches
            if verbose:
              print('[%d, %5d] loss: %.3f' % (epoch, i + 1, print_sum_loss / print_amt_losses))
            print_sum_loss = 0.0
            print_amt_losses = 0
    losses.append(sum_loss/amt_losses)
    # getting validation results
    val_loss = custom_loss(net, criterion, val_dataloader)
    net.train()
    print("Validation loss: %f" % val_loss)
    val_losses.append(val_loss)
    if checkpoint_path:
        state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses, 'test_losses': val_losses}
        torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    scheduler.step() # decreasing learning rate
  val_accuracy = accuracy(net, val_dataloader)
  print("Test accuracy: %f" % val_accuracy)
  state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses, 'test_losses': val_losses, 'test_accuracy': val_accuracy}
  if output_file:
      torch.save(state, output_file)
  return state

def accuracy(net, dataloader):
  net.to(device)
  net.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch[0].to(device), batch[1].to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total

def custom_loss(net, criterion, dataloader):
  net.to(device)
  net.eval()
  total_loss = 0
  total = 0
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch[0].to(device), batch[1].to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += 1
          total_loss += criterion(outputs, labels).item()
  return total_loss/total

def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')

# lossDict is a dictionary of keys that correspond to labels and values that are arrays
def plot(loss_dict, title, out_file, x_axis=None, x_label='Epochs', y_label='Losses'):
    # Plot the lines
    for key, value in loss_dict.items():
        if x_axis:
           plt.plot(x_axis, value, label=key)
        else:
           plt.plot(value, label=key)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + ' Losses Over Epochs')
    # Add legend
    plt.legend()
    plt.savefig(out_file)
    # Show the plot
    plt.show()


class Abs(nn.Module):
    def __init__(self, ratio=1):
        super().__init__()
        self.ratio = ratio
    def forward(self, input):
        # this is in place, might cause issues?
        input[input<0] *= -self.ratio
        return input
# Define a function to replace ReLU with the new activation function
def replace_relu_with(model, old_activation, new_activation):
    for child_name, child in model.named_children():
        if isinstance(child, old_activation):
            setattr(model, child_name, new_activation)
        else:
            replace_relu_with(child, old_activation, new_activation)
output_path = "res/part_b/resnet18/cifar10/"
data = get_data(augmentation=1)
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model.fc = nn.Linear(512, 10)

# state = train(model, data['train'], data['test'], epochs=20, lr=.01, print_every=100, output_file=output_path + "relu_model.pkl")

# # print("Testing accuracy: %f" % accuracy(model, data['test']))
# lossDict = {
#     'relu trn': state['losses'],
#     'relu test': state['test_losses']
# }
# plot(lossDict, "resnet18 cifar10 relu", output_path + "relu_model.png")

relu_state = torch.load(output_path + "relu_model.pkl", map_location=torch.device('cpu'))
old_activation = nn.ReLU
abs_state = copy.deepcopy(relu_state)
abs_state['losses'] = [abs_state['losses'][-1]]
abs_state['test_losses'] = [abs_state['test_losses'][-1]]
abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
abs_model.fc = nn.Linear(512, 10)
percent_abs_list = [0]
for i in range(1, 101):
    print("Current percent abs: ", i)
    ratio = i/100
    percent_abs_list.append(i)
    #percent_abs_list.append(ratio * 100)
    #percent_abs_list.append(ratio * 100)
    #percent_abs_list.append(ratio * 100)
    #percent_abs_list.append(ratio * 100) # doing thrice for three epochs
    # new abs activation
    new_abs = Abs(ratio=ratio)
    replace_relu_with(abs_model, old_activation, new_abs)
    old_activation = Abs  # should this be new_abs?
    abs_state['epoch'] = 0
    if i == 100:
       abs_state = train(abs_model, data['train'], data['test'], epochs=1, gamma=.85, lr=.001, print_every=100, state=abs_state, output_file=output_path + "abs_model.pkl")
    else:
       abs_state = train(abs_model, data['train'], data['test'], epochs=1, gamma=.85, lr=.001, print_every=100, state=abs_state)
loss_dict = {
   'abs trn': abs_state['losses'],
   'abs test': abs_state['test_losses']
}
plot(loss_dict, "resnet18 cifar10 abs percent", output_path + "abs_percent.png", percent_abs_list, x_label="Percent abs")

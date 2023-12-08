import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim

def train(net, device, train_dataloader, val_dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005,
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
    val_loss = custom_loss(net, device, criterion, val_dataloader)
    net.train()
    print("Validation loss: %f" % val_loss)
    val_losses.append(val_loss)
    if checkpoint_path:
        state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses, 'test_losses': val_losses}
        torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    scheduler.step() # decreasing learning rate
  val_accuracy = accuracy(net, device, val_dataloader)
  print("Test accuracy: %f" % val_accuracy)
  state = {'epoch': epoch+1, 'model': net, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses, 'test_losses': val_losses, 'test_accuracy': val_accuracy}
  if output_file:
      torch.save(state, output_file)
  return state

def accuracy(net, device, dataloader):
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

def custom_loss(net, device, criterion, dataloader):
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
def plot(loss_dict, title, out_file, x_axis=None, x_label=None, y_label=None):
    # Plot the lines
    print(loss_dict)
    for key, value in loss_dict.items():
        print(value)
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

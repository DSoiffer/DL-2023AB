import torch
import copy
import time
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def abs_activation(x, ratio=1):
    x[x<0] *= -ratio #torch.abs(x)
    return x

class Abs(nn.Module):
    def __init__(self, ratio=1):
        super().__init__()
        self.ratio = ratio
    def forward(self, input: Tensor) -> Tensor:
        # this is in place, might cause issues?
        input[input<0] *= -self.ratio
        return input

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.min_state_dict = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = 0
    total_loss = 0
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
        total_loss += loss.item()
        size += 1

        # if batch % (150*64/batch_size) == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss/size

    

# This function adapted from PyTorch's "Getting Started" examples
def test_loop(dataloader, model, loss_fn, show_acc=True):
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
            if show_acc:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    if show_acc:
        correct /= size
        print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return test_loss, correct
    else: 
        print(f"Test Error: Avg loss: {test_loss:>8f}")
        return test_loss

def run_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, hyperparameters, show_acc=True):
    epochs, batch_size, patience, min_delta = hyperparameters
    st = time.time()
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        train_losses.append(train_loss)
        if show_acc:
          val_loss, val_accuracy = test_loop(val_dataloader, model, loss_fn, show_acc=show_acc)
          val_accuracies.append(val_accuracy)
        else:
          val_loss = test_loop(val_dataloader, model, loss_fn, show_acc=show_acc)
        val_losses.append(val_loss)
        if early_stopper.early_stop(val_loss, model):             
          model.load_state_dict(early_stopper.min_state_dict)
          break
    et = time.time()
    elapsed_time = et - st
    res = {
        'epochs': e+1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'elapsed_time': elapsed_time
    }
    if show_acc:
      res['val_accuracies'] = val_accuracies
    return res

class PyTorchClassifier(BaseEstimator):
    def __init__(self, model, batch_size, learning_rate, alpha, epochs, momentum, patience, min_delta):
        #self.model = model(input_size, hidden_size, output_size)
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.momentum = momentum
        self.patience = patience
        self.min_delta = min_delta

    def fit(self, X, y):
        # Create a DataLoader for batching
        X_tensor = torch.Tensor(X)
        y_tensor = torch.Tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha, momentum=self.momentum)
        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels.long())
                loss.backward()
                optimizer.step()

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.Tensor(X)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()
    
def hyper_tuning(model, train_data, val_data, loss_fn, hyperparameters, show_acc=True):
    batch_sizes, learning_rates, alphas, epochs = hyperparameters
    best_CE = None
    best_res = None
    best_set = None

    for b in batch_sizes:
        print("Batch Size: ", b)
        train_dataloader = DataLoader(train_data, batch_size=b)
        val_dataloader = DataLoader(val_data, batch_size=b)
        for l in learning_rates:
            print("\tLearning Rate: ", l)
            for a in alphas:
                print("\t\tAlpha: ", a)
                earlyStopped = False
                for e in epochs:
                    if not earlyStopped:
                        print("\t\t\tEpochs: ", e)
                        current = {'batch_size': b, 'learning_rate': l, 'alpha': a, 'epochs': e}
                        optimizer = torch.optim.SGD(model.parameters(), lr=l, weight_decay=a, momentum=.5) #TODO momentum hyperparameter?
                        # TODO auto defining patience and min delta
                        modelRes = run_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, (e, b, 3, .1), show_acc)
                        epochs_ran = modelRes['epochs']
                        loss = modelRes['val_losses'][-1]
                        if best_CE is None or best_CE > loss: # found better model
                            best_CE = loss
                            if epochs_ran < e: # if performs better at an earlier epoch, start early
                                earlyStopped = True
                                current['epochs'] = epochs_ran
                            best_set = current
                            best_res = modelRes
    return best_res, best_set

# lossDict is a dictionary of keys that correspond to labels and values that are arrays
def plot(lossDict, title, out_file_format):
    # Plot the lines
    for key, value in lossDict.items():
        plt.plot(value, label=key)

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title(title + ' Losses Over Epochs')
    # Add legend
    plt.legend()
    plt.savefig(out_file_format +".png")
    # Show the plot
    plt.show()
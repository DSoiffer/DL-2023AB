import torch
import copy
import time

def abs_activation(x):
    return torch.abs(x)

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
        return test_loss, correct
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss
def runModel(model, train_dataloader, val_dataloader, optimizer, loss_fn, showAcc, hyperparameters):
    epochs, batch_size, patience, min_delta = hyperparameters
    st = time.time()
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        traj_params = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        if showAcc:
          val_loss, val_accuracy = test_loop(val_dataloader, model, loss_fn, showAcc=showAcc)
        else:
          val_loss = test_loop(val_dataloader, model, loss_fn, showAcc=showAcc)
        if early_stopper.early_stop(val_loss, model):             
          model.load_state_dict(early_stopper.min_state_dict)
          break
    res = (e+1,)
    res += (val_loss,)
    if showAcc:
      res += (val_accuracy,)
    et = time.time()
    elapsed_time = et - st
    res += (elapsed_time,)
    return res

def hyper_tuning(model, train_dataloader, val_dataloader, loss_fn, hyperparameters):
    batch_sizes, learning_rates, alphas, epochs = hyperparameters
    best_CE = None
    best_optimizer = None
    best_set = None
    best_acc = None
    best_time = None

    for b in batch_sizes:
        print("Batch Size: ", b)
        for l in learning_rates:
            print("\tLearning Rate: ", l)
            for a in alphas:
                print("\t\t\tAlpha: ", a)
                earlyStopped = False
                for e in epochs:
                    if not earlyStopped:
                        print("\t\tEpochs: ", e)
                        current = [b, l, a, e]
                        optimizer = torch.optim.SGD(model.parameters(), lr=l, weight_decay=a, momentum=.5) #TODO momentum hyperparameter?
                        epochs_ran, loss, acc, time = runModel(model, train_dataloader, val_dataloader, optimizer, loss_fn, True, (e, b, 3, .1))
                        if best_CE is None or best_CE > loss:
                            best_CE = loss
                            if epochs_ran < e:
                                earlyStopped = True
                                current[3] = epochs_ran
                            best_optimizer = optimizer
                            best_set = current
                            best_acc = acc
                            best_time = time
                        
    return best_CE, best_optimizer, best_set, best_acc, best_time
# def runModel(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, showAcc, hyperparameters):
#     epochs, batch_size, patience, min_delta = hyperparameters
#     st = time.time()
#     early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
#     for e in range(epochs):
#         print(f"Epoch {e+1}\n-------------------------------")
#         traj_params = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
#         if showAcc:
#           val_loss, _ = test_loop(val_dataloader, model, loss_fn, showAcc=showAcc)
#         else:
#           val_loss = test_loop(val_dataloader, model, loss_fn, showAcc=showAcc)
#         if early_stopper.early_stop(val_loss, model):             
#           model.load_state_dict(early_stopper.min_state_dict)
#           break
#     res = (e+1,)
#     if showAcc:
#       test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, showAcc=showAcc)
#       res += (test_loss,)
#       res += (test_accuracy,)
#     else:
#         test_loss = test_loop(test_dataloader, model, loss_fn, showAcc=showAcc)
#         res += (test_loss,)
#     et = time.time()
#     elapsed_time = et - st
#     res += (elapsed_time,)
#     return res
   

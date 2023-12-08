import torch.nn.utils.prune as prune
import torch.nn as nn
import copy

from shared import train, plot


def prune_model(device, model, state, data, prune_amt=0.03, iterations=5, min_acc=90):
  state['losses'] = [state['losses'][-1]] # only want one value
  state['test_losses'] = [state['test_losses'][-1]] # only want one value
  print("Min accuracy", min_acc)
  model.load_state_dict(state['net'])
  for i in range(iterations):
    print("Pruning iteration", i+1)
    l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)]
    for module in l:
        prune.l1_unstructured(module, name="weight", amount=prune_amt)
    state['epoch'] = 0
    model.load_state_dict(state['net'])
    state = train(model, device, data['train'], data['test'], epochs=2, lr=.02, gamma=.5, print_every=100)
    state['losses'] = state['losses'][:-2] # TODO won't have state data in the end due to not passing in
    state['test_losses'] = state['test_losses'][:-2]
    if state['test_accuracy'] < min_acc:
        print("Under min accuracy!")
        break
  return state

def prune_models(device, relu_model, relu_state, abs_model, abs_state, data, title, output_file, prune_amt=0.03, iterations=5, min_relu_acc=90, min_abs_acc=90):
   print("Pruning relu...")
   relu_state = prune_model(device, relu_model, relu_state, data, prune_amt, iterations, min_relu_acc)
   print("Pruning abs...")
   abs_state = prune_model(device, abs_model, abs_state, data, prune_amt, iterations, min_abs_acc)
   plot_prune(relu_state, abs_state, title, output_file)

def plot_prune(relu_state, abs_state, title, output_file):
   loss_dict = {
      'relu trn': relu_state['losses'],
      'relu test': relu_state['test_losses'],
      'abs trn': abs_state['losses'],
      'abs test': abs_state['test_losses']
   }
   plot(loss_dict, title, output_file, x_label="Percent abs")

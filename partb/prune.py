import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import copy

from shared import train, plot, custom_loss, accuracy


def prune_model(device, model, state, data, prune_amt=0.03, iterations=5, min_acc=90):
  state['losses'] = [state['losses'][-1]] # only want one value
  state['test_losses'] = [state['test_losses'][-1]] # only want one value
  print("Min accuracy", min_acc)
  model.load_state_dict(state['net'])
  print("Validation loss: %f" % custom_loss(model, device, nn.CrossEntropyLoss(), data['test']))

  for i in range(iterations):
    print("Pruning iteration", i+1)
    l = [(module, "weight") for module in model.modules() if
         isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
    l2 = [(module, "bias") for module in model.modules() if
          isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
    # for name, param in model.named_parameters():
    #     print(name)

    prune.global_unstructured(
        l+l2,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    state['epoch'] = 0
    state = train(model, device, data['train'], data['test'], epochs=2, lr=.0005, gamma=.5, print_every=100)
    state['losses'] = state['losses'][:-2] # TODO won't have state data in the end due to not passing in
    state['test_losses'] = state['test_losses'][:-2]

    a = accuracy(model, device, data['test'])
    print(accuracy(model, device, data['train']))
    print(a)

    # [print("Sparsity in " + name + " : {:.2f}%".format(
    #         100. * float(torch.sum(module.weight == 0))
    #         / float(module.weight.nelement())))
    #     for (name, module) in model.named_modules() if
    #     isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear)]
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

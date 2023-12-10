import torch.nn as nn

from shared import train, plot, replace_relu_with, Abs

def convert_0to100(abs_model, device, abs_state, data, title, output_path):
  old_activation = nn.ReLU

  new_abs = Abs(ratio=1)
  replace_relu_with(abs_model, old_activation, new_abs)
  old_activation = Abs  # should this be new_abs?
  # resetting vars that can impact training
  abs_state['epoch'] = 0
  abs_state['optimizer']['param_groups'][0]['lr']=0.01
  abs_model.load_state_dict(abs_state['net'])

  abs_state = train(abs_model, device, data['train'], data['test'], epochs=100, gamma=.9, lr=.001, print_every=100, output_file=output_path + "abs_model.pkl")

  losses = abs_state['losses']
  test_losses = abs_state['test_losses']
  loss_dict = {
    'abs trn': losses,
    'abs test': test_losses
  }
  plot(loss_dict, title, output_path + "abs_percent.png", x_label="Epochs")
  return abs_state

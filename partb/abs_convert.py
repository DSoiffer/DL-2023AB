import torch.nn as nn

from shared import train, plot, replace_relu_with, Abs

def abs_convert(abs_model, device, abs_state, data, title, output_path):
  old_activation = nn.ReLU
  abs_state['losses'] = [abs_state['losses'][-1]]
  abs_state['test_losses'] = [abs_state['test_losses'][-1]]
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
        abs_state = train(abs_model, device, data['train'], data['test'], epochs=1, gamma=.85, lr=.001, print_every=100, state=abs_state, output_file=output_path + "abs_model.pkl")
      else:
        abs_state = train(abs_model, device, data['train'], data['test'], epochs=1, gamma=.85, lr=.001, print_every=100, state=abs_state)
  loss_dict = {
    'abs trn': abs_state['losses'],
    'abs test': abs_state['test_losses']
  }
  plot(loss_dict, title, output_path + "abs_percent.png", percent_abs_list, x_label="Percent abs")
  return abs_state
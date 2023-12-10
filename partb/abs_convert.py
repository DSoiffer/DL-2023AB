import torch.nn as nn

from shared import train, plot, replace_relu_with, Abs

def abs_convert(abs_model, device, abs_state, data, title, output_path):
  old_activation = nn.ReLU
  losses = [abs_state['losses'][-1]]
  test_losses = [abs_state['test_losses'][-1]]
  percent_abs_list = [0]
  range_list = [*range(1, 101, 1)]
  range_list.insert(0,0.8)
  range_list.insert(0,0.6)
  range_list.insert(0,0.4)
  range_list.insert(0,0.2)
  range_list.insert(0, 0.1)
  range_list.insert(0,0.075)
  range_list.insert(0, 0.06256)
  range_list.insert(0, 0.05)
  range_list.insert(0,.025)
  for i in range_list:
      print("Current percent abs: ", i)
      ratio = i
      # for j in range(0, 2):
      percent_abs_list.append(i)
      # new abs activation
      new_abs = Abs(ratio=ratio)
      replace_relu_with(abs_model, old_activation, new_abs)
      old_activation = Abs  # should this be new_abs?
      # resetting vars that can impact training
      abs_state['epoch'] = 0
      abs_state['optimizer']['param_groups'][0]['lr']=0.01
      abs_model.load_state_dict(abs_state['net'])
      if i == 100:
        abs_state = train(abs_model, device, data['train'], data['test'], epochs=2, gamma=.9, lr=.001, print_every=100, output_file=output_path + "abs_model.pkl")
      else:
        abs_state = train(abs_model, device, data['train'], data['test'], epochs=2, gamma=.9, lr=.001, print_every=100)
      losses += [abs_state['losses'][-1]]
      test_losses += [abs_state['test_losses'][-1]]
  loss_dict = {
    'abs trn': losses,
    'abs test': test_losses
  }
  plot(loss_dict, title, output_path + "abs_percent.png", percent_abs_list, x_label="Percent abs")
  return abs_state

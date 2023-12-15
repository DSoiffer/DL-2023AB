from shared import train, plot

def relu_fine_tune(model, device, data, title, output_path):
  state = train(model, device, data['train'], data['test'], epochs=20, lr=.01, print_every=100, output_file=output_path + "relu_model.pkl")

  # print("Testing accuracy: %f" % accuracy(model, data['test']))
  lossDict = {
      'relu trn': state['losses'],
      'relu test': state['test_losses']
  }
  plot(lossDict, title, output_path + "relu_model.png")
  return state
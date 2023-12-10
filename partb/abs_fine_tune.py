
from shared import train, plot

def abs_fine_tune(model, device, data, title, output_path):
    state = train(model, device, data['train'], data['test'], epochs=100, lr=.001, print_every=100, output_file=output_path + "abs_model2.pkl")
    # print("Testing accuracy: %f" % accuracy(model, data['test']))
    lossDict = {
        'abs trn': state['losses'],
        'abs test': state['test_losses']
    }
    plot(lossDict, title, output_path + "abs_model2.png")
    return state

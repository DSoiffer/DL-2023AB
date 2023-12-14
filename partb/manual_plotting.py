import matplotlib.pyplot as plt
import torch
from torch import nn


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
    plt.title(title)
    # Add legend
    plt.legend()
    plt.savefig(out_file)
    # Show the plot
    plt.show()


pruneNaiveAccRelu = [x*100 for x in [.9604, .9598, .9588, .9539, .9342, .8923, .7476, .4049]]
pruneNaiveAccAbs = [x*100 for x in [.852, .8442, .8188, .7247, .5847, .4002]]
plotDictNaive = {"ReLU": pruneNaiveAccRelu, "Abs": pruneNaiveAccAbs}
plot(plotDictNaive, "Naive Pruning (20% per iteration)", "../res/part_b/resnet18/cifar10/naivepruning.png", y_label="Test Accuracy (%)", x_label="Pruning iterations")

relufile = open('../res/part_b/resnet18/cifar10/prunereluonly.txt', 'r')
pruneAccRelu = [96.05]
pruneLossRelu = [.1469603205641991]
lines = relufile.readlines()
for line in lines:
    parts = line.strip().split(' ')
    if parts[0] == "Test":
        pruneAccRelu.append(float(parts[2])*100)
    elif parts[0] == "Validation":
        pruneLossRelu.append(float(parts[2]))

absfile = open('../res/part_b/resnet18/cifar10/pruneabsonly.txt', 'r')
pruneAccAbs = [85.31]
pruneLossAbs = [.6707804625547384]
lines = absfile.readlines()
for line in lines:
    parts = line.strip().split(' ')
    if parts[0] == "Test":
        pruneAccAbs.append(float(parts[2])*100)
    elif parts[0] == "Validation":
        pruneLossAbs.append(float(parts[2]))

plotDictAcc = {"ReLU": pruneAccRelu, "Abs": pruneAccAbs}
plot(plotDictAcc, "Pruning/Tuning (20% per iteration, 3 epochs)", "../res/part_b/resnet18/cifar10/pruningacc.png", y_label="Test Accuracy (%)", x_label="Pruning iterations")

plotDictLoss = {"ReLU": pruneLossRelu, "Abs": pruneLossAbs}
plot(plotDictLoss, "Pruning/Tuning (20% per iteration, 3 epochs)", "../res/part_b/resnet18/cifar10/pruningloss.png", y_label="Loss", x_label="Epochs (3/iteration)")



absSparsitiesFile = open('../res/part_b/resnet18/cifar10/aaa.txt', 'r')
lines = absSparsitiesFile.readlines()
sparsitiesAbs = []
for line in lines:
    parts = line.strip().split(' ')
    sparsitiesAbs.append(float(parts[4].replace('%', '')))

reluSparsitiesFile = open('../res/part_b/resnet18/cifar10/bbb.txt', 'r')
lines = reluSparsitiesFile.readlines()
sparsitiesRelu = []
for line in lines:
    parts = line.strip().split(' ')
    sparsitiesRelu.append(float(parts[4].replace('%', '')))


abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
abs_model.fc = nn.Linear(512, 10)
abs_state = torch.load('../res/part_b/resnet18/cifar10/' + "abs_model2.pkl", map_location=torch.device('cuda'))
abs_model = abs_state['model']
print(abs_model)
l = [name for (name, module) in abs_model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear)]

for a in range(len(sparsitiesRelu)):
    print(l[a], sparsitiesAbs[a] - sparsitiesRelu[a])
#load resnet
#load imagenet
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import data_loader.FashionMNIST as FashionMNIST_loader
import networks.FashionMNIST as FashionMNIST_networks
import torchvision.transforms as transforms
from setup import Abs, runModel

#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# print(model)

# for p in model.parameters():
#     print(p)



# Define your new activation function
new_activation = Abs(ratio=0) #nn.Tanh()  

# Define a function to replace ReLU with the new activation function
def replace_relu_with(model, old_activation, new_activation):
    for child_name, child in model.named_children():
        if isinstance(child, old_activation):
            setattr(model, child_name, new_activation)
        else:
            replace_relu_with(child, old_activation, new_activation)

# Replace ReLU with the new activation function in your model
#replace_relu_with(model, nn.ReLU, new_activation)

#print(model)


# def abs_activation(x, ratio=1):
#     return x[x<0] * -ratio #torch.abs(x)

#load data
train_data, val_data, test_data = FashionMNIST_loader.load(True)
train_dataloader = DataLoader(train_data, batch_size=64) 
val_dataloader = DataLoader(val_data, batch_size=64)

# "alpha": 0.0001,
#         "batch_size": 64,
#         "epochs": 50,
#         "learning_rate": 0.01,
#         "momentum": 0.5
model = FashionMNIST_networks.Deep(nn.ReLU())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01/50, weight_decay=0.0001, momentum=.5)
runModel(model, train_dataloader, val_dataloader, optimizer, nn.CrossEntropyLoss(), (50, 64, 3, .1))

#training loop
old_activation = nn.ReLU
for i in range(1, 101):

    ratio = i/100
    # this might be a problem if it can't differentiate this lambda?
    new_abs = Abs(ratio=ratio)
    replace_relu_with(model, old_activation, new_abs)
    old_activation = Abs  # should this be new_abs?
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01/50, weight_decay=0.0001, momentum=.5)
    runModel(model, train_dataloader, val_dataloader, optimizer, nn.CrossEntropyLoss(), (5, 64, 3, .1))
    #train until roughly converged



# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])

#transform=transform
full_dataset = datasets.ImageNet(root='C:\\Users\\dunca\\Downloads\\imagenet\\', split='val'  )

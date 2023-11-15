#load resnet
#load imagenet
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from setup import Abs

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

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
replace_relu_with(model, nn.ReLU, new_activation)

#print(model)


# def abs_activation(x, ratio=1):
#     return x[x<0] * -ratio #torch.abs(x)

#training loop
old_activation = nn.ReLU
for i in range(1, 101):

    ratio = i/100
    # this might be a problem if it can't differentiate this lambda?
    new_abs = Abs(ratio=ratio)
    replace_relu_with(model, old_activation, new_abs)
    old_activation = Abs
    #train until roughly converged
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageNet(root='/path/to/imagenet/train', split='train', transform=transform, download=True)

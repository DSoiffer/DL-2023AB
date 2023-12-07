import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
import torch.nn as nn
import copy

from shared import replace_relu_with, Abs
from relu_fine_tune import relu_fine_tune
from prune import prune_models
from abs_convert import abs_convert


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
print(device)

def get_data(augmentation=0):
  # Data augmentation transformations. Not for Testing!
  if augmentation:
    transform_train = transforms.Compose([
      transforms.Resize(128),
      transforms.RandomCrop(128, padding=8, padding_mode='edge'), # Take 128x128 crops from 136x136 padded images
      transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
      transforms.ToTensor(),
    ])
  else:
    transform_train = transforms.ToTensor()

  transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
  ])

  trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)

  # generator = torch.Generator().manual_seed(42)
  # trainset, valset = random_split(trainset, [.8, .2], generator=generator)
  # trainset.transform = transform_train
  # valset.transform = transform_test
  # print(trainset, valset)
  # # print(trainset.transform)
  # print(valset.transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                            num_workers=0)
  # valloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
  #                                           num_workers=2)

  testset = datasets.CIFAR10(root='../data', train=False, download=True,
                                      transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                          num_workers=0)
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  return {'train': trainloader, 'test': testloader, 'classes': classes}

# beginning values
output_path = "../res/part_b/resnet18/cifar10/"
data = get_data(augmentation=1)

# fine tuning relu
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# model.fc = nn.Linear(512, 10)
# relu_state = relu_fine_tune(model, device, data, "resnet18 cifar10 relu", output_path)

# converting to abs
# relu_state = torch.load(output_path + "relu_model.pkl", map_location=torch.device('cuda'))
# abs_state = copy.deepcopy(relu_state)
# abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# abs_model.fc = nn.Linear(512, 10)
# abs_convert(abs_model, device, abs_state, data, "resnet18 cifar10 abs percent", output_path)

# pruning relu/abs
relu_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
relu_model.fc = nn.Linear(512, 10)
relu_state = torch.load(output_path + "relu_model.pkl", map_location=torch.device('cpu'))
abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
abs_model.fc = nn.Linear(512, 10)
abs_state = torch.load(output_path + "abs_model.pkl", map_location=torch.device('cpu'))
replace_relu_with(abs_model, nn.ReLU, Abs(ratio=1))
min_relu_acc = relu_state['test_accuracy'] - .01
min_abs_acc = abs_state['test_accuracy'] - .01
prune_models(device, relu_model, relu_state, abs_model, abs_state, data, "Losses after successive pruning", output_path + "pruning.png", min_relu_acc=min_relu_acc, min_abs_acc=min_abs_acc, iterations=10)
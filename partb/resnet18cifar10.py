import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
import torch.nn as nn
import copy

from partb.convert_0to100 import convert_0to100
from shared import replace_relu_with, Abs, train, accuracy, custom_loss
from relu_fine_tune import relu_fine_tune
from abs_fine_tune import abs_fine_tune
from prune import prune_models, prune_model
from abs_convert import abs_convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#    device = torch.device("mps")
print(device)



def get_data(augmentation=0):
    # Data augmentation transformations. Not for Testing!
    if augmentation:
        transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(128, padding=8, padding_mode='edge'),  # Take 128x128 crops from 136x136 padded images
            transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
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

def get_data_half(augmentation=0):
    # Data augmentation transformations. Not for Testing!
    if augmentation:
        transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(128, padding=8, padding_mode='edge'),  # Take 128x128 crops from 136x136 padded images
            transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
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
def fine_tune_relu():
    print("Fine tuning relu_state...")
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(512, 10)
    new_relu_state = relu_fine_tune(model, device, data, "resnet18 cifar10 relu", output_path)
    torch.save(new_relu_state, output_path + "relu_model_NEW.pkl")
#fine_tune_relu()


# converting to abs
def convert_to_abs():
    relu_state = torch.load(output_path + "relu_model.pkl", map_location=torch.device('cuda'))
    abs_state = copy.deepcopy(relu_state)
    abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    abs_model.fc = nn.Linear(512, 10)
    abs_convert(abs_model, device, abs_state, data, "resnet18 cifar10 abs percent", output_path)
# convert_to_abs()

'''
# training final abs
# abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# abs_model.fc = nn.Linear(512, 10)
abs_state = torch.load(output_path + "abs_model.pkl", map_location=torch.device('cuda'))
abs_model = abs_state['model']
#replace_relu_with(abs_model, nn.ReLU, Abs(ratio=1))
#replace_relu_with(abs_model, Abs, nn.ReLU())
#print(abs_model)
abs_fine_tune(abs_model, device, data, "resnet18 cifar10 abs", output_path)
'''



# pruning relu/abs
relu_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
relu_model.fc = nn.Linear(512, 10)
relu_state = torch.load(output_path + "relu_model.pkl", map_location=torch.device('cuda'))
relu_state['epoch'] = 0
relu_model.load_state_dict(relu_state['net'])
# relu_state = train(relu_model, device, data['train'], data['test'], epochs=1, gamma=.85, lr=.001, print_every=100, state=relu_state)
abs_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
abs_model.fc = nn.Linear(512, 10)
abs_state = torch.load(output_path + "abs_model2.pkl", map_location=torch.device('cuda'))
abs_model = abs_state['model']
# # replace_relu_with(abs_model, nn.ReLU, Abs(ratio=1))
min_relu_acc = relu_state['test_accuracy'] - .03
min_abs_acc = abs_state['test_accuracy'] - .03
print("About to start pruning")
# prune_models(device, relu_model, relu_state, abs_model, abs_state, data, "Losses after successive pruning",
#              output_path + "pruning.png", prune_amt=0.1, min_relu_acc=min_relu_acc, min_abs_acc=min_abs_acc,
#              iterations=30)
#abs_state = convert_0to100(relu_model, device, relu_state, data, "Converting from ReLU to Abs At Once", output_path+"convertAllAtOnce")
#torch.save(abs_state, output_path + "abs_model_0to100AtOnce.pkl")

# relu_model.load_state_dict(relu_state['net'])
# print("Testing metrics:")
print(accuracy(abs_model, device, data['test']))
print(custom_loss(abs_model, device, nn.CrossEntropyLoss(), data['test']))
print(accuracy(relu_model, device, data['test']))
print(custom_loss(relu_model, device, nn.CrossEntropyLoss(), data['test']))
# # print(custom_loss(relu_model, device, nn.CrossEntropyLoss(), data['test']))
#
# prune_amt = 0.1
# iterations = 10
# relu_state = prune_model(device, relu_model, relu_state, data, prune_amt, iterations, min_relu_acc)

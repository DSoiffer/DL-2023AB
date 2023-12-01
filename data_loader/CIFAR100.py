import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

def load(validation = True):
  training_data = datasets.CIFAR100(
    root="data/CIFAR100",
    train=True,
    download=True,
    transform=ToTensor()
  )
  test_data = datasets.CIFAR100(
      root="data/CIFAR100",
      train=False,
      download=True,
      transform=ToTensor()
  )

  # Split into training and validation sets
  if validation:
    training_data, validation_data = torch.utils.data.random_split(training_data, [.85, .15])
    return training_data, validation_data, test_data
  else:
    return training_data, test_data

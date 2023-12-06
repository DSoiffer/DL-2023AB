import torch
from torchvision import datasets
import torchvision.transforms as transforms
def load(validation = True):
  training_data = datasets.CIFAR100(
    root="data/CIFAR100",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  )
  test_data = datasets.CIFAR100(
      root="data/CIFAR100",
      train=False,
      download=True,
      transform= transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
  )

  # Split into training and validation sets
  if validation:
    training_data, validation_data = torch.utils.data.random_split(training_data, [.85, .15])
    return training_data, validation_data, test_data
  else:
    return training_data, test_data

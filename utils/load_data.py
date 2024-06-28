import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10
from PIL import Image
import random

def load_data(data, bs):

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    if data == 'cifar10':
        train_dataset = datasets.CIFAR10('../Dataset/cifar10/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../Dataset/cifar10/', train=False, download=True, transform=transform)

    elif data == 'stl10':
        train_dataset = datasets.STL10('../Dataset/stl10', split="train", download=True, transform=transform)
        test_dataset = datasets.STL10('../Dataset/stl10', split="test", download=True, transform=transform)

    print('Train dataset: %d, Test dataset: %d'%(len(train_dataset),len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle = False)

    return train_loader,  test_loader


def normalzie(args, x):

    if args.dataset == 'cifar10':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'stl10':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)


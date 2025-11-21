import os
import torch
from torchvision import datasets, transforms

def get_cifar10(batch_size: int, num_workers: int = 4):
    cifar_root = os.environ.get("CIFAR10_ROOT", "../data")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_set = datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, 
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir, batch_size=32):
    train_tf, test_tf = get_transforms()

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_tf)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(train_dataset.classes)
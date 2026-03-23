"""
CIFAR-10 Dataset Loader
Handles loading and preprocessing of CIFAR-10 data
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=64, data_dir='./data', download=True):
    """
    Get CIFAR-10 train and test data loaders.
    
    CIFAR-10:
    - 60,000 32x32 color images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 training + 10,000 test
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        download: Whether to download the dataset if it is missing
        
    Returns:
        train_loader, test_loader
    """
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load datasets
    try:
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform_test
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"CIFAR-10 dataset is unavailable in '{data_dir}'. "
            f"Set download=True with network access or place the dataset there first. "
            f"Original error: {exc}"
        ) from exc
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar10_classes():
    """Get CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def denormalize_cifar10(tensor):
    """
    Denormalize CIFAR-10 tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    return tensor * std + mean

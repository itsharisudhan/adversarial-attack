"""
MNIST Dataset Loader
Handles loading and preprocessing of MNIST data
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64, data_dir='./data', download=True):
    """
    Get MNIST train and test data loaders.
    
    MNIST:
    - 70,000 28x28 grayscale images
    - 10 classes: digits 0-9
    - 60,000 training + 10,000 test
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        download: Whether to download the dataset if it is missing
        
    Returns:
        train_loader, test_loader
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load datasets
    try:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=transform
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"MNIST dataset is unavailable in '{data_dir}'. "
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

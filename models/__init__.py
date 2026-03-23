"""
Models Module
"""

from .simple_cnn import SimpleCNN_MNIST, SimpleCNN_CIFAR10, train_model, evaluate_model
from .resnet import ResNet18, resnet18_cifar10
from .vgg import VGG11, vgg11_cifar10

__all__ = [
    'SimpleCNN_MNIST',
    'SimpleCNN_CIFAR10',
    'ResNet18',
    'VGG11',
    'resnet18_cifar10',
    'vgg11_cifar10',
    'train_model',
    'evaluate_model'
]

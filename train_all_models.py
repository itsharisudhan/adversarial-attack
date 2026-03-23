"""
Train All Models Script
Trains MNIST and CIFAR-10 models for the 60% milestone demo.
"""

import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import get_cifar10_loaders, get_mnist_loaders
from models import SimpleCNN_CIFAR10, SimpleCNN_MNIST


def train_model(model, train_loader, test_loader, epochs, lr, device, model_name):
    """Train one model and save the best checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{train_loss / (pbar.n + 1):.4f}",
                    "acc": f"{100.0 * train_correct / train_total:.2f}%",
                }
            )

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * test_correct / test_total
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Accuracy: {100.0 * train_correct / train_total:.2f}%")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            os.makedirs("trained_models", exist_ok=True)
            torch.save(model.state_dict(), f"trained_models/{model_name}.pth")
            print(f"  [OK] Saved best model (Accuracy: {best_accuracy:.2f}%)")

    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
    return model


def require_dataset(name, loader_fn, batch_size):
    """Load a dataset or exit with a clear setup message."""
    print(f"\nLoading {name} dataset...")
    try:
        train_loader, test_loader = loader_fn(batch_size=batch_size)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("Training requires network access the first time datasets are downloaded.")
        sys.exit(1)

    print(f"[OK] Loaded {len(train_loader.dataset)} training samples")
    print(f"[OK] Loaded {len(test_loader.dataset)} test samples")
    return train_loader, test_loader


def main():
    """Main training function."""
    print("=" * 60)
    print("ADVERSARIAL DETECTION SYSTEM - MODEL TRAINING")
    print("60% Milestone Implementation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        "mnist": {"epochs": 1, "batch_size": 64, "lr": 0.001},
        "cifar10": {"epochs": 1, "batch_size": 128, "lr": 0.001},
    }

    print("\n" + "=" * 60)
    print("1. TRAINING MNIST MODEL")
    print("=" * 60)
    mnist_train_loader, mnist_test_loader = require_dataset(
        "MNIST", get_mnist_loaders, batch_size=config["mnist"]["batch_size"]
    )
    train_model(
        model=SimpleCNN_MNIST(),
        train_loader=mnist_train_loader,
        test_loader=mnist_test_loader,
        epochs=config["mnist"]["epochs"],
        lr=config["mnist"]["lr"],
        device=device,
        model_name="mnist_cnn",
    )

    print("\n" + "=" * 60)
    print("2. TRAINING CIFAR-10 SIMPLE CNN")
    print("=" * 60)
    cifar_train_loader, cifar_test_loader = require_dataset(
        "CIFAR-10", get_cifar10_loaders, batch_size=config["cifar10"]["batch_size"]
    )
    train_model(
        model=SimpleCNN_CIFAR10(),
        train_loader=cifar_train_loader,
        test_loader=cifar_test_loader,
        epochs=config["cifar10"]["epochs"],
        lr=config["cifar10"]["lr"],
        device=device,
        model_name="cifar10_cnn",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nTrained Models:")
    print("  [OK] trained_models/mnist_cnn.pth")
    print("  [OK] trained_models/cifar10_cnn.pth")
    print("\nYou can now:")
    print("  1. Run the demo: python demo_60.py")
    print("  2. Start the web interface: python web/app.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

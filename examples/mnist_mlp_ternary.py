#!/usr/bin/env python3
"""MNIST MLP training with TernaryLinear layers.

This example demonstrates ternary quantization on a real dataset (MNIST),
comparing TernaryLinearCUDA vs FP16 baseline in terms of accuracy and speed.

Usage:
    uv run python examples/mnist_mlp_ternary.py
    uv run python examples/mnist_mlp_ternary.py --epochs 5
    uv run python examples/mnist_mlp_ternary.py --no-ternary  # FP16 only baseline
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    """Simple 3-layer MLP for MNIST.

    Architecture: 784 → 256 → 128 → 10
    """

    def __init__(self, use_ternary: bool = True):
        super().__init__()

        if use_ternary:
            from bittorch.nn import TernaryLinearCUDA
            Linear = TernaryLinearCUDA
        else:
            Linear = nn.Linear

        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_mnist_loaders(
    batch_size: int = 128, data_dir: str = "./data"
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders.

    Args:
        batch_size: Batch size for data loaders.
        data_dir: Directory to download/load MNIST data.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module, test_loader: DataLoader, device: str
) -> tuple[float, float]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        device: Device to evaluate on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    use_ternary: bool = True,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    data_dir: str = "./data",
) -> dict:
    """Train MNIST model and return results.

    Args:
        use_ternary: If True, use TernaryLinearCUDA. If False, use nn.Linear.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed.
        data_dir: Directory for MNIST data.

    Returns:
        Dictionary with training results.
    """
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_ternary and device != "cuda":
        raise RuntimeError("TernaryLinearCUDA requires CUDA. Use --no-ternary for CPU.")

    model_name = "TernaryLinearCUDA" if use_ternary else "nn.Linear (FP32)"

    print(f"\n{'='*60}")
    print(f"Training {model_name} on MNIST")
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size, data_dir)

    # Model
    model = MNISTNet(use_ternary=use_ternary).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Training loop
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Train: {train_acc:.2%} ({train_loss:.4f}) | "
            f"Test: {test_acc:.2%} ({test_loss:.4f}) | "
            f"Time: {epoch_time:.1f}s"
        )

    # Final summary
    print(f"\n{model_name} Final Results:")
    print(f"  Test Accuracy:  {history['test_acc'][-1]:.2%}")
    print(f"  Test Loss:      {history['test_loss'][-1]:.4f}")
    print(f"  Avg Epoch Time: {sum(history['epoch_time'])/len(history['epoch_time']):.1f}s")

    return {
        "model_name": model_name,
        "use_ternary": use_ternary,
        "final_test_acc": history["test_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "history": history,
        "model": model,
    }


def print_quantized_weights_summary(model: nn.Module) -> None:
    """Print summary of quantized weights."""
    print("\n--- Quantized Weights Summary ---")
    for name, module in model.named_modules():
        if hasattr(module, "get_quantized_weight"):
            w_tern, scale = module.get_quantized_weight()
            zeros = (w_tern == 0).float().mean().item()
            ones = (w_tern.abs() == 1).float().mean().item()
            print(f"{name}:")
            print(f"  Shape: {tuple(w_tern.shape)}")
            print(f"  Sparsity (zeros): {zeros:.1%}")
            print(f"  Non-zeros: {ones:.1%}")
            print(f"  Scale range: [{scale.min().item():.4f}, {scale.max().item():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP with TernaryLinear")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--no-ternary", action="store_true", help="Use FP32 nn.Linear instead of ternary"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare ternary vs FP32"
    )
    args = parser.parse_args()

    if args.compare:
        # Run both and compare
        print("\n" + "=" * 60)
        print("COMPARISON: TernaryLinearCUDA vs FP32 nn.Linear")
        print("=" * 60)

        # Train ternary
        ternary_results = train_model(
            use_ternary=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        print_quantized_weights_summary(ternary_results["model"])

        # Train FP32
        fp32_results = train_model(
            use_ternary=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
        )

        # Summary comparison
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} | {'Ternary':>12} | {'FP32':>12} | {'Diff':>10}")
        print("-" * 60)

        t_acc = ternary_results["final_test_acc"]
        f_acc = fp32_results["final_test_acc"]
        print(f"{'Test Accuracy':<20} | {t_acc:>11.2%} | {f_acc:>11.2%} | {(t_acc-f_acc)*100:>+9.2f}%")

        t_loss = ternary_results["final_test_loss"]
        f_loss = fp32_results["final_test_loss"]
        print(f"{'Test Loss':<20} | {t_loss:>12.4f} | {f_loss:>12.4f} | {t_loss-f_loss:>+10.4f}")

        t_time = sum(ternary_results["history"]["epoch_time"]) / args.epochs
        f_time = sum(fp32_results["history"]["epoch_time"]) / args.epochs
        print(f"{'Avg Epoch Time':<20} | {t_time:>11.1f}s | {f_time:>11.1f}s | {t_time-f_time:>+9.1f}s")

    else:
        # Single run
        results = train_model(
            use_ternary=not args.no_ternary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
        )

        if not args.no_ternary:
            print_quantized_weights_summary(results["model"])


if __name__ == "__main__":
    main()

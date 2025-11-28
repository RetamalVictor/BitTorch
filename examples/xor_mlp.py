#!/usr/bin/env python3
"""XOR MLP training with TernaryLinear layers.

This example demonstrates that ternary quantization + STE + optimizer
can converge on a trivial problem (XOR truth table).

Usage:
    uv run python examples/xor_mlp.py
    uv run python examples/xor_mlp.py --cuda   # Use CUDA-accelerated backend
"""

import argparse

import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear


class XORNet(nn.Module):
    """Simple 2-layer MLP for XOR problem.

    Architecture: 2 → 4 → 1
    """

    def __init__(self, use_ternary: bool = True, use_cuda_backend: bool = False):
        super().__init__()

        if use_ternary:
            if use_cuda_backend:
                from bittorch.nn import TernaryLinearCUDA
                Linear = TernaryLinearCUDA
            else:
                Linear = TernaryLinear
        else:
            Linear = nn.Linear

        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_xor_data(
    n_per_quadrant: int = 250,
    noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a training-ready XOR dataset with many samples around each corner.

    Args:
        n_per_quadrant: Number of samples to generate for each of the four XOR input regions.
        noise: Standard deviation of Gaussian noise added around each XOR corner.

    Returns:
        A tuple (inputs, targets) where:
            inputs: Tensor of shape (4 * n_per_quadrant, 2) containing noisy samples
                    around the four canonical XOR input points.
            targets: Tensor of shape (4 * n_per_quadrant, 1) with corresponding XOR labels.
    """
    centers = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    labels = torch.tensor([0., 1., 1., 0.], dtype=torch.float32)

    idx = torch.repeat_interleave(torch.arange(4), n_per_quadrant)
    base = centers[idx]
    jitter = torch.randn_like(base) * noise

    inputs = base + jitter
    targets = labels[idx].unsqueeze(1)

    return inputs, targets

def train_xor(
    use_ternary: bool = True,
    use_cuda_backend: bool = False,
    num_steps: int = 2000,
    lr: float = 0.01,
    print_every: int = 200,
    seed: int = 42,
) -> tuple[XORNet, list[float]]:
    """Train XOR network.

    Args:
        use_ternary: If True, use TernaryLinear. If False, use nn.Linear.
        use_cuda_backend: If True, use TernaryLinearCUDA (requires CUDA).
        num_steps: Number of training steps.
        lr: Learning rate.
        print_every: Print loss every N steps.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trained_model, loss_history)
    """
    torch.manual_seed(seed)

    device = "cuda" if use_cuda_backend else "cpu"

    # Create model and data
    model = XORNet(use_ternary=use_ternary, use_cuda_backend=use_cuda_backend)
    model = model.to(device)
    inputs, targets = get_xor_data()
    inputs, targets = inputs.to(device), targets.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    loss_history = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (step + 1) % print_every == 0:
            print(f"Step {step + 1:4d} | Loss: {loss.item():.6f}")

    return model, loss_history


def evaluate_xor(
    model: XORNet, n_per_quadrant: int = 250, noise: float = 0.1, device: str = "cpu"
) -> dict:
    """Evaluate model on XOR dataset and return statistics.

    Args:
        model: Trained XORNet model.
        n_per_quadrant: Number of samples per quadrant for evaluation.
        noise: Noise level for sample generation.
        device: Device to run evaluation on.

    Returns:
        Dictionary with evaluation statistics.
    """
    inputs, targets = get_xor_data(n_per_quadrant=n_per_quadrant, noise=noise)
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

    # Overall metrics
    correct = (preds == targets).sum().item()
    total = len(targets)
    accuracy = correct / total

    # Per-quadrant analysis
    samples_per_q = n_per_quadrant
    quadrant_stats = []
    quadrant_labels = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]

    for q in range(4):
        start, end = q * samples_per_q, (q + 1) * samples_per_q
        q_preds = preds[start:end]
        q_targets = targets[start:end]
        q_probs = probs[start:end]

        q_correct = (q_preds == q_targets).sum().item()
        q_acc = q_correct / samples_per_q
        q_mean_prob = q_probs.mean().item()
        q_std_prob = q_probs.std().item()

        quadrant_stats.append({
            "label": quadrant_labels[q],
            "accuracy": q_acc,
            "mean_prob": q_mean_prob,
            "std_prob": q_std_prob,
        })

    # Print summary
    print(f"\nXOR Evaluation ({total} samples, noise={noise}):")
    print("-" * 50)
    print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("\nPer-quadrant breakdown:")
    print(f"{'Quadrant':<12} | {'Acc':>6} | {'Mean Prob':>10} | {'Std':>6}")
    print("-" * 50)
    for s in quadrant_stats:
        print(f"{s['label']:<12} | {s['accuracy']:>5.1%} | {s['mean_prob']:>10.4f} | {s['std_prob']:>6.4f}")
    print("-" * 50)

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "quadrants": quadrant_stats,
    }


def main():
    """Main function to run XOR training experiment."""
    parser = argparse.ArgumentParser(description="XOR MLP Training with TernaryLinear")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA-accelerated TernaryLinearCUDA"
    )
    parser.add_argument(
        "--steps", type=int, default=2000, help="Number of training steps"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    use_cuda = args.cuda
    device = "cuda" if use_cuda else "cpu"

    if use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        use_cuda = False
        device = "cpu"

    backend_name = "TernaryLinearCUDA" if use_cuda else "TernaryLinear"

    print("=" * 50)
    print(f"XOR MLP Training with {backend_name}")
    print(f"Device: {device}")
    print("=" * 50)

    # Train with ternary weights
    print(f"\n--- Training with {backend_name} ---")
    model_ternary, losses_ternary = train_xor(
        use_ternary=True,
        use_cuda_backend=use_cuda,
        num_steps=args.steps,
        lr=args.lr,
    )
    evaluate_xor(model_ternary, device=device)

    # Compare with standard Linear (optional)
    print("\n--- Training with nn.Linear (baseline) ---")
    model_fp, losses_fp = train_xor(
        use_ternary=False,
        use_cuda_backend=False,  # nn.Linear doesn't need CUDA backend flag
        num_steps=args.steps,
        lr=args.lr,
    )
    # Keep baseline on CPU for simplicity
    evaluate_xor(model_fp, device="cpu")

    # Print quantized weights for inspection
    print(f"\n--- Quantized Weights ({backend_name}) ---")
    for name, module in model_ternary.named_modules():
        if hasattr(module, "get_quantized_weight"):
            w_tern, scale = module.get_quantized_weight()
            print(f"\n{name}:")
            print(f"  w_tern:\n{w_tern.cpu()}")
            print(f"  scale: {scale.cpu()}")

    # Final comparison
    print("\n--- Final Loss Comparison ---")
    print(f"{backend_name} final loss: {losses_ternary[-1]:.6f}")
    print(f"nn.Linear final loss:     {losses_fp[-1]:.6f}")


if __name__ == "__main__":
    main()

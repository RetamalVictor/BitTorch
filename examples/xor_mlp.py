#!/usr/bin/env python3
"""XOR MLP training with TernaryLinear layers.

This example demonstrates that ternary quantization + STE + optimizer
can converge on a trivial problem (XOR truth table).

Usage:
    uv run python examples/xor_mlp.py
"""

import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear


class XORNet(nn.Module):
    """Simple 2-layer MLP for XOR problem.

    Architecture: 2 → 4 → 1
    """

    def __init__(self, use_ternary: bool = True):
        super().__init__()
        Linear = TernaryLinear if use_ternary else nn.Linear

        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Get XOR truth table as tensors.

    Returns:
        Tuple of (inputs, targets) where:
            inputs: shape (4, 2) - all XOR input combinations
            targets: shape (4, 1) - XOR outputs
    """
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return inputs, targets


def train_xor(
    use_ternary: bool = True,
    num_steps: int = 2000,
    lr: float = 0.01,
    print_every: int = 200,
    seed: int = 42,
) -> tuple[XORNet, list[float]]:
    """Train XOR network.

    Args:
        use_ternary: If True, use TernaryLinear. If False, use nn.Linear.
        num_steps: Number of training steps.
        lr: Learning rate.
        print_every: Print loss every N steps.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trained_model, loss_history)
    """
    torch.manual_seed(seed)

    # Create model and data
    model = XORNet(use_ternary=use_ternary)
    inputs, targets = get_xor_data()

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


def evaluate_xor(model: XORNet) -> None:
    """Evaluate model on XOR truth table and print results."""
    inputs, targets = get_xor_data()

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)

    print("\nXOR Evaluation:")
    print("-" * 40)
    print("Input     | Target | Prediction | Correct")
    print("-" * 40)

    correct = 0
    for i in range(4):
        inp = inputs[i].tolist()
        target = targets[i].item()
        pred = predictions[i].item()
        pred_class = 1.0 if pred > 0.5 else 0.0
        is_correct = pred_class == target
        correct += is_correct

        print(f"{inp[0]:.0f}, {inp[1]:.0f}    |  {target:.0f}     |   {pred:.4f}   | {'✓' if is_correct else '✗'}")

    print("-" * 40)
    print(f"Accuracy: {correct}/4 ({100 * correct / 4:.0f}%)")


def main():
    """Main function to run XOR training experiment."""
    print("=" * 50)
    print("XOR MLP Training with TernaryLinear")
    print("=" * 50)

    # Train with ternary weights
    print("\n--- Training with TernaryLinear ---")
    model_ternary, losses_ternary = train_xor(use_ternary=True, num_steps=2000, lr=0.01)
    evaluate_xor(model_ternary)

    # Compare with standard Linear (optional)
    print("\n--- Training with nn.Linear (baseline) ---")
    model_fp, losses_fp = train_xor(use_ternary=False, num_steps=2000, lr=0.01)
    evaluate_xor(model_fp)

    # Print quantized weights for inspection
    print("\n--- Quantized Weights (TernaryLinear) ---")
    for name, module in model_ternary.named_modules():
        if isinstance(module, TernaryLinear):
            w_tern, scale = module.get_quantized_weight()
            print(f"\n{name}:")
            print(f"  w_tern:\n{w_tern}")
            print(f"  scale: {scale}")

    # Final comparison
    print("\n--- Final Loss Comparison ---")
    print(f"TernaryLinear final loss: {losses_ternary[-1]:.6f}")
    print(f"nn.Linear final loss:     {losses_fp[-1]:.6f}")


if __name__ == "__main__":
    main()

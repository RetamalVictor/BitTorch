"""Tests for XOR MLP training with TernaryLinear.

This serves as a regression test to ensure ternary quantization + STE
can still converge on a trivial problem after any changes.
"""

import pytest
import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear


class XORNet(nn.Module):
    """Simple 2-layer MLP for XOR problem."""

    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(2, 4)
        self.fc2 = TernaryLinear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_xor_data():
    """Get XOR truth table."""
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return inputs, targets


class TestXORMLP:
    """Tests for XOR MLP training."""

    def test_xor_convergence(self):
        """TernaryLinear MLP should solve XOR problem."""
        torch.manual_seed(42)

        model = XORNet()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        # Train for enough steps
        for _ in range(2000):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()

        # Should achieve 100% accuracy
        accuracy = (predictions == targets).float().mean().item()
        assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy * 100}%"

    def test_xor_loss_decreases(self):
        """Loss should decrease during training."""
        torch.manual_seed(42)

        model = XORNet()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        # Get initial loss
        with torch.no_grad():
            initial_loss = criterion(model(inputs), targets).item()

        # Train
        for _ in range(1000):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Get final loss
        with torch.no_grad():
            final_loss = criterion(model(inputs), targets).item()

        # Loss should have decreased significantly
        assert final_loss < initial_loss * 0.1, (
            f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_xor_weights_are_ternary(self):
        """After training, quantized weights should be ternary."""
        torch.manual_seed(42)

        model = XORNet()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        # Train briefly
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Check weights are ternary
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                w_tern, scale = module.get_quantized_weight()
                unique_values = torch.unique(w_tern)

                for val in unique_values:
                    assert val.item() in [-1.0, 0.0, 1.0], (
                        f"Non-ternary value found: {val.item()}"
                    )

    def test_xor_gradients_flow(self):
        """Gradients should flow through the entire network."""
        torch.manual_seed(42)

        model = XORNet()
        inputs, targets = get_xor_data()

        criterion = nn.BCEWithLogitsLoss()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_xor_no_nan_outputs(self):
        """Outputs should never be NaN during training."""
        torch.manual_seed(42)

        model = XORNet()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        for step in range(500):
            optimizer.zero_grad()
            outputs = model(inputs)

            assert not torch.isnan(outputs).any(), f"NaN output at step {step}"

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

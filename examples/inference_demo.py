#!/usr/bin/env python
"""Demonstration of packed ternary inference.

This example shows:
1. Creating a model with TernaryLinear (for training)
2. Converting to TernaryLinearInference (for deployment)
3. Comparing parameter memory usage
4. Comparing runtime memory usage (CUDA only)
5. Running inference with packed weights

Usage:
    uv run python examples/inference_demo.py
    uv run python examples/inference_demo.py --cuda
    uv run python examples/inference_demo.py --cuda --hidden 1024
"""

import argparse
import gc

import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear, TernaryLinearInference
from bittorch.utils import (
    convert_linear_layers,
    ternary_linear_to_infer,
    get_model_memory_bytes,
    count_ternary_params,
)


class TernaryMLP(nn.Module):
    """Simple MLP with TernaryLinear layers for demonstration."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = TernaryLinear(in_features, hidden, bias=True)
        self.fc2 = TernaryLinear(hidden, hidden, bias=True)
        self.fc3 = TernaryLinear(hidden, out_features, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_fp32_mlp(in_features: int, hidden: int, out_features: int) -> nn.Module:
    """Create equivalent FP32 model for comparison."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


def main():
    parser = argparse.ArgumentParser(description="Packed ternary inference demo")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hidden dim: {args.hidden}")
    print()

    # Create models
    in_features = 256
    out_features = 64

    print("=" * 60)
    print("1. Model Creation")
    print("=" * 60)

    # FP32 baseline
    fp32_model = create_fp32_mlp(in_features, args.hidden, out_features).to(device)
    fp32_mem = get_model_memory_bytes(fp32_model)
    print(f"FP32 model memory: {fp32_mem['total']:,} bytes ({fp32_mem['total']/1024:.1f} KB)")

    # Ternary training model
    ternary_model = TernaryMLP(in_features, args.hidden, out_features).to(device)
    ternary_train_mem = get_model_memory_bytes(ternary_model)
    print(f"Ternary training memory: {ternary_train_mem['total']:,} bytes ({ternary_train_mem['total']/1024:.1f} KB)")
    print("  (Same as FP32 - stores master weights for training)")

    print()
    print("=" * 60)
    print("2. Convert to Inference Mode")
    print("=" * 60)

    # Convert layer by layer
    infer_fc1 = ternary_linear_to_infer(ternary_model.fc1)
    infer_fc2 = ternary_linear_to_infer(ternary_model.fc2)
    infer_fc3 = ternary_linear_to_infer(ternary_model.fc3)

    # Build inference model
    class InferenceMLP(nn.Module):
        def __init__(self, fc1, fc2, fc3):
            super().__init__()
            self.fc1 = fc1
            self.fc2 = fc2
            self.fc3 = fc3
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    infer_model = InferenceMLP(infer_fc1, infer_fc2, infer_fc3).to(device)
    infer_mem = get_model_memory_bytes(infer_model)
    print(f"Inference model memory: {infer_mem['total']:,} bytes ({infer_mem['total']/1024:.1f} KB)")
    print(f"  Packed ternary: {infer_mem['packed_ternary']:,} bytes")
    print(f"  Scales: {infer_mem['scales']:,} bytes")

    reduction = fp32_mem['total'] / infer_mem['total']
    print(f"\nMemory reduction: {reduction:.1f}x vs FP32")

    print()
    print("=" * 60)
    print("3. Verify Numerical Equivalence")
    print("=" * 60)

    # Test with same input
    torch.manual_seed(42)
    x = torch.randn(args.batch, in_features, device=device)

    ternary_model.eval()
    with torch.no_grad():
        y_train = ternary_model(x)
        y_infer = infer_model(x)

    diff = (y_train - y_infer).abs().max().item()
    print(f"Max difference between training and inference: {diff:.2e}")
    print(f"Outputs match: {torch.allclose(y_train, y_infer, rtol=1e-4, atol=1e-4)}")

    print()
    print("=" * 60)
    print("4. Alternative: convert_linear_layers()")
    print("=" * 60)

    # Can also use convert_linear_layers for any nn.Sequential model
    fp_model = create_fp32_mlp(in_features, args.hidden, out_features)
    converted_model = convert_linear_layers(fp_model)

    converted_mem = get_model_memory_bytes(converted_model)
    print(f"Converted model memory: {converted_mem['total']:,} bytes")

    # Count by type
    counts = count_ternary_params(converted_model)
    print(f"Parameter counts: {counts}")

    # Runtime memory measurement (CUDA only)
    if device == "cuda":
        print()
        print("=" * 60)
        print("5. Runtime Memory Usage (CUDA)")
        print("=" * 60)

        def measure_forward_memory(model, x, name):
            """Measure peak GPU memory during forward pass."""
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated()
            print(f"{name}: {peak:,} bytes ({peak/1024/1024:.2f} MB)")
            return peak

        # Measure each model
        x_cuda = torch.randn(args.batch, in_features, device='cuda')
        fp32_runtime = measure_forward_memory(fp32_model, x_cuda, "FP32 forward")
        train_runtime = measure_forward_memory(ternary_model, x_cuda, "Ternary training forward")
        infer_runtime = measure_forward_memory(infer_model, x_cuda, "Ternary inference forward")

        print()
        print("Runtime memory comparison:")
        print(f"  Inference vs FP32: {fp32_runtime/infer_runtime:.1f}x less")
        print(f"  Inference vs Training: {train_runtime/infer_runtime:.1f}x less")
        print()
        print("Note: The packed CUDA kernel reads 2-bit weights directly,")
        print("never allocating a full [N, K] weight tensor in GPU memory.")

    print()
    print("=" * 60)
    print("6. Summary")
    print("=" * 60)

    print(f"""
    Model Type              | Param Mem (KB) | Reduction
    ------------------------|----------------|----------
    FP32 baseline           | {fp32_mem['total']/1024:>12.1f} | 1.0x
    Ternary (training)      | {ternary_train_mem['total']/1024:>12.1f} | 1.0x
    Ternary (inference)     | {infer_mem['total']/1024:>12.1f} | {reduction:.1f}x

    The packed inference model achieves {reduction:.1f}x parameter memory
    reduction while producing numerically equivalent outputs.

    On CUDA, the packed kernel also reduces runtime memory by avoiding
    the allocation of full weight tensors during forward pass.

    For deployment on memory-constrained devices (like Jetson),
    use TernaryLinearInference after training with TernaryLinear.

    For detailed benchmarks, run:
        uv run python benchmark/bench_packed_inference.py --cuda
    """)


if __name__ == "__main__":
    main()

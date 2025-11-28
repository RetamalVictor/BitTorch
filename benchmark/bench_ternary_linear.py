#!/usr/bin/env python3
"""Benchmark script for TernaryLinear vs nn.Linear.

Usage:
    python benchmark/bench_ternary_linear.py [options]

Options:
    --batch-size INT       Batch size (default: 32)
    --in-features INT      Input features (default: 512)
    --out-features INT     Output features (default: 512)
    --warmup INT           Warmup iterations (default: 10)
    --iterations INT       Benchmark iterations (default: 100)
    --device STR           Device: cpu or cuda (default: cpu)
    --include-backward     Include backward pass in timing

Environment variables:
    BITTORCH_DEVICE        Override default device
    BITTORCH_BATCH_SIZE    Override default batch size
"""

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_size: int = 32
    in_features: int = 512
    out_features: int = 512
    warmup: int = 10
    iterations: int = 100
    device: str = "cpu"
    include_backward: bool = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int


def time_forward(layer: nn.Module, x: torch.Tensor, iterations: int) -> list[float]:
    """Time forward pass only."""
    times = []
    for _ in range(iterations):
        if x.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = layer(x)

        if x.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return times


def time_forward_backward(
    layer: nn.Module, x: torch.Tensor, iterations: int
) -> list[float]:
    """Time forward + backward pass."""
    times = []
    for _ in range(iterations):
        if x.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # Forward
        y = layer(x)
        loss = y.sum()

        # Backward
        loss.backward()

        if x.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        # Zero gradients for next iteration
        layer.zero_grad()

    return times


def run_benchmark(
    layer: nn.Module, x: torch.Tensor, config: BenchmarkConfig, name: str
) -> BenchmarkResult:
    """Run benchmark for a single layer."""
    # Warmup
    timing_fn = time_forward_backward if config.include_backward else time_forward
    _ = timing_fn(layer, x, config.warmup)

    # Benchmark
    times = timing_fn(layer, x, config.iterations)

    times_tensor = torch.tensor(times)
    return BenchmarkResult(
        name=name,
        mean_ms=times_tensor.mean().item(),
        std_ms=times_tensor.std().item(),
        min_ms=times_tensor.min().item(),
        max_ms=times_tensor.max().item(),
        iterations=config.iterations,
    )


def print_result(result: BenchmarkResult) -> None:
    """Print a single benchmark result."""
    print(f"  {result.name}:")
    print(f"    Mean: {result.mean_ms:.4f} ms (+/- {result.std_ms:.4f})")
    print(f"    Min:  {result.min_ms:.4f} ms")
    print(f"    Max:  {result.max_ms:.4f} ms")


def print_comparison(ternary: BenchmarkResult, linear: BenchmarkResult) -> None:
    """Print comparison between two results."""
    ratio = ternary.mean_ms / linear.mean_ms

    print(f"\n  Ratio (TernaryLinear / nn.Linear): {ratio:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TernaryLinear vs nn.Linear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BITTORCH_BATCH_SIZE", 32)),
        help="Batch size",
    )
    parser.add_argument(
        "--in-features", type=int, default=512, help="Input features"
    )
    parser.add_argument(
        "--out-features", type=int, default=512, help="Output features"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("BITTORCH_DEVICE", "cpu"),
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Include backward pass in timing",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        include_backward=args.include_backward,
    )

    # Print configuration
    print("=" * 60)
    print("BitTorch Benchmark: TernaryLinear vs nn.Linear")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device:           {config.device}")
    print(f"  Batch size:       {config.batch_size}")
    print(f"  In features:      {config.in_features}")
    print(f"  Out features:     {config.out_features}")
    print(f"  Warmup:           {config.warmup} iterations")
    print(f"  Iterations:       {config.iterations}")
    print(f"  Include backward: {config.include_backward}")

    # Create layers
    torch.manual_seed(42)
    device = torch.device(config.device)

    ternary_layer = TernaryLinear(
        config.in_features, config.out_features
    ).to(device)
    linear_layer = nn.Linear(config.in_features, config.out_features).to(device)

    # Copy weights for fair comparison
    with torch.no_grad():
        linear_layer.weight.copy_(ternary_layer.weight)
        linear_layer.bias.copy_(ternary_layer.bias)

    # Create input
    x = torch.randn(config.batch_size, config.in_features, device=device)

    # Run benchmarks
    mode = "Forward" if not config.include_backward else "Forward + Backward"
    print(f"\n{mode} Pass Timing:")
    print("-" * 40)

    ternary_result = run_benchmark(ternary_layer, x, config, "TernaryLinear")
    print_result(ternary_result)

    linear_result = run_benchmark(linear_layer, x, config, "nn.Linear")
    print_result(linear_result)

    print_comparison(ternary_result, linear_result)

    # Additional size benchmarks
    sizes = [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    print(f"\n\nScaling with Matrix Size (batch={config.batch_size}):")
    print("-" * 60)
    print(f"{'Size':>12} | {'TernaryLinear':>14} | {'nn.Linear':>14} | {'Ratio':>8}")
    print("-" * 60)

    for in_f, out_f in sizes:
        try:
            ternary = TernaryLinear(in_f, out_f).to(device)
            linear = nn.Linear(in_f, out_f).to(device)

            with torch.no_grad():
                linear.weight.copy_(ternary.weight)
                linear.bias.copy_(ternary.bias)

            x_sized = torch.randn(config.batch_size, in_f, device=device)

            # Quick benchmark (fewer iterations)
            quick_config = BenchmarkConfig(
                batch_size=config.batch_size,
                in_features=in_f,
                out_features=out_f,
                warmup=5,
                iterations=50,
                device=config.device,
                include_backward=config.include_backward,
            )

            t_result = run_benchmark(ternary, x_sized, quick_config, "ternary")
            l_result = run_benchmark(linear, x_sized, quick_config, "linear")

            ratio = t_result.mean_ms / l_result.mean_ms
            print(
                f"{in_f}x{out_f:>5} | {t_result.mean_ms:>11.4f} ms | "
                f"{l_result.mean_ms:>11.4f} ms | {ratio:>7.2f}x"
            )
        except RuntimeError:
            print(f"{in_f}x{out_f:>5} | {'OOM':>14} | {'OOM':>14} | {'N/A':>8}")

    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Comprehensive benchmark: TernaryLinear vs nn.Linear (FP32/FP16).

Compares:
- nn.Linear FP32
- nn.Linear FP16
- TernaryLinear (pure Python)
- TernaryLinearCUDA

Usage:
    uv run python benchmark/bench_ternary_vs_linear.py [options]

Options:
    --device STR       Device: cuda (default) or cpu
    --warmup INT       Warmup iterations (default: 10)
    --iterations INT   Benchmark iterations (default: 100)
    --shapes STR       Shape preset: small, medium, large, all (default: all)

Environment variables:
    BITTORCH_DEVICE    Override default device
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: float  # samples/sec


# Standard shapes from roadmap
SHAPES = {
    "small": [(32, 256, 256)],
    "medium": [(64, 1024, 4096)],
    "large": [(16, 4096, 4096)],
    "all": [
        (32, 256, 256),
        (64, 1024, 4096),
        (16, 4096, 4096),
    ],
}


def time_function(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    """Time a function with proper CUDA sync."""
    # Warmup
    for _ in range(warmup):
        _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = fn()

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return times


def time_forward_backward(
    layer: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    """Time forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        y = layer(x)
        y.sum().backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)
        layer.zero_grad()

    return times


def benchmark_layer(
    layer: nn.Module,
    x: torch.Tensor,
    name: str,
    warmup: int,
    iterations: int,
    device: torch.device,
    include_backward: bool = False,
) -> BenchmarkResult:
    """Benchmark a layer and return results."""
    batch_size = x.shape[0]

    if include_backward:
        times = time_forward_backward(layer, x, warmup, iterations, device)
    else:
        times = time_function(
            lambda: layer(x), warmup, iterations, device
        )

    times_t = torch.tensor(times)
    mean_ms = times_t.mean().item()
    throughput = (batch_size / mean_ms) * 1000  # samples/sec

    return BenchmarkResult(
        name=name,
        mean_ms=mean_ms,
        std_ms=times_t.std().item(),
        min_ms=times_t.min().item(),
        max_ms=times_t.max().item(),
        throughput=throughput,
    )


def print_table_header() -> None:
    """Print benchmark table header."""
    print(f"{'Layer':<25} | {'Mean (ms)':>10} | {'Std':>8} | {'Throughput':>12} | {'vs FP32':>8}")
    print("-" * 75)


def print_result(result: BenchmarkResult, baseline_ms: float | None = None) -> None:
    """Print a benchmark result row."""
    ratio = f"{result.mean_ms / baseline_ms:.2f}x" if baseline_ms else "1.00x"
    print(
        f"{result.name:<25} | {result.mean_ms:>10.4f} | {result.std_ms:>8.4f} | "
        f"{result.throughput:>10.0f}/s | {ratio:>8}"
    )


def run_shape_benchmark(
    batch: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    include_backward: bool,
) -> dict[str, BenchmarkResult]:
    """Run benchmark for a single shape configuration."""
    from bittorch.nn import TernaryLinear

    results = {}

    # Create input
    x_fp32 = torch.randn(batch, in_features, device=device)
    x_fp16 = x_fp32.half() if device.type == "cuda" else x_fp32

    # 1. nn.Linear FP32
    linear_fp32 = nn.Linear(in_features, out_features).to(device)
    results["nn.Linear FP32"] = benchmark_layer(
        linear_fp32, x_fp32, "nn.Linear FP32",
        warmup, iterations, device, include_backward
    )

    # 2. nn.Linear FP16 (CUDA only)
    if device.type == "cuda":
        linear_fp16 = nn.Linear(in_features, out_features).to(device).half()
        results["nn.Linear FP16"] = benchmark_layer(
            linear_fp16, x_fp16, "nn.Linear FP16",
            warmup, iterations, device, include_backward
        )

    # 3. TernaryLinear (pure Python - force no CUDA kernel)
    ternary_python = TernaryLinear(
        in_features, out_features, use_cuda_kernel=False
    ).to(device)
    results["TernaryLinear (Python)"] = benchmark_layer(
        ternary_python, x_fp32, "TernaryLinear (Python)",
        warmup, iterations, device, include_backward
    )

    # 4. TernaryLinear with CUDA kernel (if available)
    if device.type == "cuda":
        try:
            from bittorch.nn import TernaryLinearCUDA
            ternary_cuda = TernaryLinearCUDA(in_features, out_features).to(device)
            results["TernaryLinearCUDA"] = benchmark_layer(
                ternary_cuda, x_fp32, "TernaryLinearCUDA",
                warmup, iterations, device, include_backward
            )
        except Exception as e:
            print(f"  (TernaryLinearCUDA unavailable: {e})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TernaryLinear vs nn.Linear (FP32/FP16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("BITTORCH_DEVICE", "cuda"),
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--shapes",
        type=str,
        default="all",
        choices=["small", "medium", "large", "all"],
    )
    parser.add_argument("--backward", action="store_true", help="Include backward pass")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # Print header
    print("=" * 75)
    print("BitTorch Benchmark: TernaryLinear vs nn.Linear")
    print("=" * 75)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Mode: {'Forward + Backward' if args.backward else 'Forward only'}")

    shapes = SHAPES[args.shapes]

    for batch, in_f, out_f in shapes:
        print(f"\n{'='*75}")
        print(f"Shape: batch={batch}, in={in_f}, out={out_f}")
        print("=" * 75)
        print_table_header()

        try:
            results = run_shape_benchmark(
                batch, in_f, out_f, device,
                args.warmup, args.iterations, args.backward
            )

            baseline = results.get("nn.Linear FP32")
            baseline_ms = baseline.mean_ms if baseline else None

            for name in ["nn.Linear FP32", "nn.Linear FP16",
                         "TernaryLinear (Python)", "TernaryLinearCUDA"]:
                if name in results:
                    print_result(results[name], baseline_ms)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM for shape ({batch}, {in_f}, {out_f})")
            else:
                raise

    print("\n" + "=" * 75)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Benchmark packed ternary inference vs baselines.

This benchmark compares memory usage and latency across:
1. nn.Linear (FP32/FP16 baseline)
2. TernaryLinear (training module with FP master weights)
3. TernaryLinearInference (packed 2-bit weights with CUDA kernel)

Shapes tested:
- MLP-ish: (B=32, K=1024, N=4096)
- Transformer-ish: (B=32*512, K=768, N=768)  [batch * seq flattened]
- Small-batch: (B=1, K=768, N=768)

Usage:
    uv run python benchmark/bench_packed_inference.py
    uv run python benchmark/bench_packed_inference.py --dtype fp16
    uv run python benchmark/bench_packed_inference.py --warmup 50 --iters 200
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    shape: str
    dtype: str
    peak_memory_bytes: int
    avg_latency_ms: float
    std_latency_ms: float
    param_memory_bytes: int


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / (1024 * 1024)


def measure_peak_memory(fn: Callable, warmup: int = 10) -> int:
    """Measure peak GPU memory during function execution."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Clear and reset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure
    fn()
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated()


def measure_latency(fn: Callable, warmup: int = 10, iters: int = 100) -> tuple:
    """Measure latency statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def get_param_memory(module: nn.Module) -> int:
    """Get parameter/buffer memory in bytes."""
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    for buf in module.buffers():
        if buf is not None:
            total += buf.numel() * buf.element_size()
    return total


def benchmark_fp_linear(
    B: int, K: int, N: int,
    dtype: torch.dtype,
    warmup: int, iters: int
) -> BenchmarkResult:
    """Benchmark standard nn.Linear."""
    x = torch.randn(B, K, device='cuda', dtype=dtype)
    linear = nn.Linear(K, N, bias=False).to(device='cuda', dtype=dtype)

    def forward():
        with torch.no_grad():
            return linear(x)

    peak_mem = measure_peak_memory(forward, warmup)
    avg_lat, std_lat = measure_latency(forward, warmup, iters)
    param_mem = get_param_memory(linear)

    dtype_name = "fp16" if dtype == torch.float16 else "fp32"
    return BenchmarkResult(
        name=f"nn.Linear ({dtype_name})",
        shape=f"({B}, {K}) -> ({B}, {N})",
        dtype=dtype_name,
        peak_memory_bytes=peak_mem,
        avg_latency_ms=avg_lat,
        std_latency_ms=std_lat,
        param_memory_bytes=param_mem,
    )


def benchmark_ternary_training(
    B: int, K: int, N: int,
    dtype: torch.dtype,
    warmup: int, iters: int
) -> BenchmarkResult:
    """Benchmark TernaryLinear (training module)."""
    from bittorch.nn import TernaryLinear

    x = torch.randn(B, K, device='cuda', dtype=dtype)
    linear = TernaryLinear(K, N, bias=False).to(device='cuda', dtype=dtype)
    linear.eval()

    def forward():
        with torch.no_grad():
            return linear(x)

    peak_mem = measure_peak_memory(forward, warmup)
    avg_lat, std_lat = measure_latency(forward, warmup, iters)
    param_mem = get_param_memory(linear)

    dtype_name = "fp16" if dtype == torch.float16 else "fp32"
    return BenchmarkResult(
        name=f"TernaryLinear ({dtype_name})",
        shape=f"({B}, {K}) -> ({B}, {N})",
        dtype=dtype_name,
        peak_memory_bytes=peak_mem,
        avg_latency_ms=avg_lat,
        std_latency_ms=std_lat,
        param_memory_bytes=param_mem,
    )


def benchmark_ternary_inference(
    B: int, K: int, N: int,
    dtype: torch.dtype,
    warmup: int, iters: int
) -> BenchmarkResult:
    """Benchmark TernaryLinearInference (packed CUDA kernel)."""
    from bittorch.nn import TernaryLinear, TernaryLinearInference
    from bittorch.utils import ternary_linear_to_infer

    # Create training module, convert to inference
    train_module = TernaryLinear(K, N, bias=False).to(device='cuda', dtype=dtype)
    infer_module = ternary_linear_to_infer(train_module).to(device='cuda')

    # Ensure scale is correct dtype
    infer_module.scale = infer_module.scale.to(dtype)

    x = torch.randn(B, K, device='cuda', dtype=dtype)

    def forward():
        with torch.no_grad():
            return infer_module(x)

    peak_mem = measure_peak_memory(forward, warmup)
    avg_lat, std_lat = measure_latency(forward, warmup, iters)
    param_mem = get_param_memory(infer_module)

    dtype_name = "fp16" if dtype == torch.float16 else "fp32"
    return BenchmarkResult(
        name=f"TernaryLinearInference ({dtype_name})",
        shape=f"({B}, {K}) -> ({B}, {N})",
        dtype=dtype_name,
        peak_memory_bytes=peak_mem,
        avg_latency_ms=avg_lat,
        std_latency_ms=std_lat,
        param_memory_bytes=param_mem,
    )


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GB"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KB"
    else:
        return f"{n} B"


def print_results_table(results: List[BenchmarkResult], title: str):
    """Print results as a formatted table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    # Header
    print(f"{'Method':<35} | {'Param Mem':>12} | {'Peak Mem':>12} | {'Latency':>15}")
    print(f"{'-' * 35}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 15}")

    # Rows
    for r in results:
        latency_str = f"{r.avg_latency_ms:.3f} +/- {r.std_latency_ms:.3f} ms"
        print(f"{r.name:<35} | {format_bytes(r.param_memory_bytes):>12} | {format_bytes(r.peak_memory_bytes):>12} | {latency_str:>15}")

    # Comparison
    if len(results) >= 2:
        print()
        baseline = results[0]
        for r in results[1:]:
            param_ratio = baseline.param_memory_bytes / max(r.param_memory_bytes, 1)
            peak_ratio = baseline.peak_memory_bytes / max(r.peak_memory_bytes, 1)
            print(f"  {r.name} vs {baseline.name}:")
            print(f"    Parameter memory: {param_ratio:.1f}x smaller")
            print(f"    Peak memory: {peak_ratio:.1f}x smaller")


def run_benchmark_suite(
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    shapes: Optional[List[tuple]] = None,
):
    """Run the full benchmark suite."""
    if shapes is None:
        shapes = [
            # (name, B, K, N)
            ("MLP-like", 32, 1024, 4096),
            ("Transformer-like", 32 * 512, 768, 768),
            ("Small-batch", 1, 768, 768),
            ("Large", 32, 4096, 4096),
        ]

    print(f"\nBenchmark Configuration:")
    print(f"  dtype: {'fp16' if dtype == torch.float16 else 'fp32'}")
    print(f"  warmup: {warmup} iterations")
    print(f"  measurement: {iters} iterations")
    print(f"  device: {torch.cuda.get_device_name()}")

    for name, B, K, N in shapes:
        results = []

        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        try:
            results.append(benchmark_fp_linear(B, K, N, dtype, warmup, iters))
        except Exception as e:
            print(f"  FP Linear failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            results.append(benchmark_ternary_training(B, K, N, dtype, warmup, iters))
        except Exception as e:
            print(f"  TernaryLinear failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            results.append(benchmark_ternary_inference(B, K, N, dtype, warmup, iters))
        except Exception as e:
            print(f"  TernaryLinearInference failed: {e}")

        if results:
            print_results_table(results, f"{name}: ({B}, {K}) -> ({B}, {N})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark packed ternary inference")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                        help="Data type for computation")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                        help="Measurement iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print("=" * 80)
    print(" BitTorch Packed Inference Benchmark")
    print("=" * 80)

    run_benchmark_suite(dtype, args.warmup, args.iters)

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    print("""
TernaryLinearInference uses a packed CUDA kernel that:
  - Stores weights as 2-bit packed uint8 (4 weights per byte)
  - Reads packed weights directly in the kernel
  - Unpacks in registers, never allocates full weight tensor
  - Uses add/sub instead of multiply for ternary {-1, 0, +1}

This provides:
  - ~16x reduction in parameter memory vs FP32
  - ~8x reduction in parameter memory vs FP16
  - Reduced memory bandwidth during inference
  - Similar compute patterns to BitNet-style ternary networks
""")


if __name__ == "__main__":
    main()

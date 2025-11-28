#!/usr/bin/env python3
"""Micro-benchmark for raw CUDA ternary kernel (no Python overhead).

Measures the kernel time alone by calling _C.ternary_linear_forward directly.

Usage:
    uv run python benchmark/bench_ternary_kernel_only.py [options]

Options:
    --warmup INT       Warmup iterations (default: 50)
    --iterations INT   Benchmark iterations (default: 500)
    --shapes STR       Shape preset: small, medium, large, all (default: all)
"""

import argparse
import time

import torch

# Check CUDA availability upfront
if not torch.cuda.is_available():
    print("ERROR: This benchmark requires CUDA.")
    exit(1)


SHAPES = [
    # (batch, in_features, out_features)
    (32, 256, 256),      # Small
    (64, 512, 512),      # Medium-small
    (64, 1024, 4096),    # Medium (transformer-like)
    (16, 4096, 4096),    # Large square
    (128, 768, 3072),    # BERT-like FFN
    (32, 4096, 11008),   # LLaMA-like FFN
]


def benchmark_kernel(
    x: torch.Tensor,
    w_tern: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Benchmark raw kernel call. Returns (mean_us, std_us)."""
    import bittorch._C as _C

    # Warmup
    for _ in range(warmup):
        _ = _C.ternary_linear_forward(x, w_tern, scale, bias)
        torch.cuda.synchronize()

    # Benchmark
    times_us = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = _C.ternary_linear_forward(x, w_tern, scale, bias)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)  # microseconds

    times_t = torch.tensor(times_us)
    return times_t.mean().item(), times_t.std().item()


def benchmark_cublas_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Benchmark cuBLAS GEMM (nn.Linear equivalent). Returns (mean_us, std_us)."""
    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.linear(x, weight, bias)
        torch.cuda.synchronize()

    # Benchmark
    times_us = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = torch.nn.functional.linear(x, weight, bias)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)

    times_t = torch.tensor(times_us)
    return times_t.mean().item(), times_t.std().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark for ternary CUDA kernel")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    # Check kernel availability
    try:
        import bittorch._C as _C
        assert hasattr(_C, "ternary_linear_forward")
    except (ImportError, AssertionError):
        print("ERROR: bittorch CUDA extension not available.")
        print("Run 'uv build' first.")
        exit(1)

    print("=" * 80)
    print("BitTorch Kernel Micro-Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    print(f"{'Shape (B,K,N)':<20} | {'Ternary (us)':>14} | {'cuBLAS (us)':>14} | {'Ratio':>8} | {'TFLOPS':>8}")
    print("-" * 80)

    for batch, in_f, out_f in SHAPES:
        # Create inputs
        x = torch.randn(batch, in_f, device="cuda")
        w_tern = torch.randint(-1, 2, (out_f, in_f), device="cuda", dtype=torch.int8)
        scale = torch.rand(out_f, device="cuda") + 0.5
        bias = torch.randn(out_f, device="cuda")

        # For cuBLAS comparison
        weight_fp32 = w_tern.float() * scale.unsqueeze(1)

        try:
            # Benchmark ternary kernel
            tern_mean, tern_std = benchmark_kernel(
                x, w_tern, scale, bias, args.warmup, args.iterations
            )

            # Benchmark cuBLAS
            cublas_mean, cublas_std = benchmark_cublas_gemm(
                x, weight_fp32, bias, args.warmup, args.iterations
            )

            ratio = tern_mean / cublas_mean

            # Compute TFLOPS (2 * B * K * N operations)
            flops = 2 * batch * in_f * out_f
            tflops = (flops / tern_mean) / 1e6  # TFLOPS from microseconds

            shape_str = f"({batch},{in_f},{out_f})"
            print(
                f"{shape_str:<20} | {tern_mean:>11.2f} +/- {tern_std:>4.1f} | "
                f"{cublas_mean:>11.2f} +/- {cublas_std:>4.1f} | {ratio:>7.2f}x | {tflops:>7.2f}"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                shape_str = f"({batch},{in_f},{out_f})"
                print(f"{shape_str:<20} | {'OOM':>14} | {'OOM':>14} | {'N/A':>8}")
            else:
                raise

    print("=" * 80)
    print("\nNotes:")
    print("- Ternary kernel: Custom CUDA kernel for ternary GEMM")
    print("- cuBLAS: PyTorch's F.linear (uses cuBLAS under the hood)")
    print("- Ratio < 1.0 means ternary is faster than cuBLAS")
    print("- Current kernel is baseline (not yet optimized with tiling)")


if __name__ == "__main__":
    main()

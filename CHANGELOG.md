# Changelog

All notable changes to BitTorch are documented here.

---

## v0.1.5 (2024-11-28)

Stability, polish, and pre-INT4 preparation release.

### New Features
- **Extensible quantization framework** (`bittorch/quant/base.py`):
  - Base classes: `QuantConfig`, `Quantizer`, `QuantType`, `ScaleType`
  - `TernaryQuantConfig` now inherits from `QuantConfig`
  - `Int4QuantConfig` placeholder showing how to extend for new formats
  - `TernaryQuantizer` class for object-oriented quantization pipelines

- **Developer documentation**:
  - `CONTRIBUTING.md`: Setup, testing, benchmarking, adding kernels, PR guidelines
  - `docs/kernel_development.md`: Tiling strategy, memory layout, profiling guide

- **Improved error handling** in CUDA dispatch (`csrc/core/dispatch.cpp`):
  - Comprehensive shape/dtype validation with helpful messages
  - Device compatibility checks
  - Automatic contiguity handling

### Stats
- 185 tests passing (unchanged from v0.1.4)
- Codebase ready for INT4 extension

---

## v0.1.4 (2024-11-28)

Extended coverage and robustness release.

### New Features
- **Character-level language model** (`examples/tiny_char_lm_ternary.py`):
  - MLP-based character prediction on Shakespeare text
  - `--compare` mode for ternary vs FP32 perplexity comparison
  - `--download` option for TinyShakespeare dataset
  - Results: FP32 PPL 1.00, Ternary PPL 1.33 (~33% gap)

- **Backend parameter** for `TernaryLinear`:
  - `backend="auto"` (default): Automatically select best backend
  - `backend="cuda"`: Force CUDA kernel (error if unavailable)
  - `backend="python"`: Force pure Python implementation
  - Deprecated `use_cuda_kernel` with warning

- **Robustness test suite** (`tests/test_robustness.py`):
  - 65 new tests covering 13 shapes on CPU and CUDA
  - 6 random seeds tested for determinism
  - Edge cases: zero input, large input, threshold weights
  - Big-shape smoke tests (32×4096×4096, etc.) marked `@pytest.mark.slow`
  - Training stability tests

### Stats
- 185 tests passing (up from 120)
- 12 slow tests (deselected by default)

---

## v0.1.3 (2024-11-28)

Tiled kernel optimization release.

### New Features
- **Tiled CUDA kernel** with shared memory optimization:
  - 32×32×32 tile sizes for batch, output, and reduction dimensions
  - Cooperative loading into shared memory
  - 1.4-1.9x speedup over baseline for medium/large shapes

- **Kernel dispatch logic** with `BITTORCH_KERNEL` environment variable:
  - `baseline`: Force baseline kernel (for debugging)
  - `tiled`: Force tiled kernel

- **Kernel regression tests** (`tests/test_kernel_regression.py`):
  - 18 tests ensuring numerical correctness between kernel variants

### Performance Results

| Shape (B,K,N) | Baseline | Tiled | Speedup | vs cuBLAS |
|---------------|----------|-------|---------|-----------|
| (32,256,256) | 97.7 μs | 101.1 μs | 0.97x | 3.6x slower |
| (64,1024,4096) | 5927 μs | 3159 μs | **1.88x** | 11.3x slower |
| (128,768,3072) | 6700 μs | 3595 μs | **1.86x** | 11.8x slower |
| (32,4096,11008) | 31725 μs | 18138 μs | **1.75x** | 10.1x slower |

### Stats
- 120 tests passing (up from 102)

---

## v0.1.2 (2024-11-28)

Benchmarks and performance characterization release.

### New Features
- `benchmark/bench_ternary_vs_linear.py`: Comprehensive benchmark comparing:
  - nn.Linear FP32 and FP16
  - TernaryLinear (Python)
  - TernaryLinearCUDA
  - Multiple shapes, forward and forward+backward timing
- `benchmark/bench_ternary_kernel_only.py`: Micro-benchmark for raw CUDA kernel

### Documentation
- README.md now includes benchmark tables with concrete numbers
- Documents baseline kernel performance (7-33x slower than cuBLAS)

### Key Findings
- Baseline ternary kernel needs optimization (v0.1.3)
- FP16 nn.Linear is 2-3x faster than FP32

---

## v0.1.1 (2025-11-28)

Gradient semantics stabilization release.

### Bug Fixes
- **Fixed gradient mismatch between Python and CUDA implementations**
  - Detached `scale` in `ternary_quantize_ste()` to prevent gradients flowing through scale computation
  - Scale is now treated as a non-differentiable calibration statistic
  - Cosine similarity between Python and CUDA weight gradients now >0.99 (was ~0.29)

### New Tests
- Added `tests/test_gradient_consistency.py` with 8 new tests:
  - Weight gradient matching (cosine similarity)
  - Weight gradient allclose
  - Bias gradient matching
  - Input gradient matching
  - Multi-seed consistency
  - Various shapes consistency
  - Gradcheck tests for autograd correctness


### Stats
- 102 tests passing (up from 94)

---

## v0.1.0 (2025-11-28)

First public release of BitTorch.

### Features
- Ternary quantization (1.58-bit weights)
- CUDA forward kernel for optimized inference
- Training support via STE + PyTorch backward
- Auto-CUDA detection in TernaryLinear
- 94 tests passing

### Examples
- XOR MLP convergence demo
- MNIST MLP with ternary vs FP32 comparison

### Benchmark (MNIST, 5 epochs)
- TernaryLinear: 94.3% accuracy
- nn.Linear FP32: 97.3% accuracy

---

## Commit History

### 0c816da - Stage 7: v0.1.0 release - polish, docs, versioning

**Date:** 2025-11-28

API improvements:
- TernaryLinear now auto-uses CUDA kernel when on GPU
- Added use_cuda_kernel parameter (default True)
- Falls back to pure Python on CPU or when kernel unavailable

Documentation:
- Added README.md with installation, quick start, benchmark table
- API reference for TernaryLinear and TernaryLinearCUDA

Versioning:
- Bumped to v0.1.0 in __init__.py and pyproject.toml

This is the first release of BitTorch with:
- Ternary quantization (1.58-bit weights)
- CUDA forward kernel
- Training via STE + PyTorch backward
- 94 tests passing
- XOR and MNIST examples

---

### 99bf284 - Stage 6: MNIST MLP with TernaryLinearCUDA

**Date:** 2025-11-28

- examples/mnist_mlp_ternary.py: 3-layer MLP (784→256→128→10) with:
  - --cuda flag for TernaryLinearCUDA
  - --compare mode for side-by-side evaluation
  - --epochs, --batch-size, --lr arguments

- tests/test_mnist_tiny_regression.py: 7 regression tests
  - Forward/backward NaN checks
  - Loss decrease verification
  - Uses synthetic data (no MNIST download in CI)

Results (5 epochs):
  - TernaryLinearCUDA: ~94-95% test accuracy
  - FP32 nn.Linear: ~97% test accuracy
  - ~3% gap acceptable for 1.58-bit quantization

Also: Added torchvision dep, data/ to .gitignore

---

### aca2967 - Stage 4 & 5: CUDA ternary GEMM kernel + TernaryLinearCUDA module

**Date:** 2025-11-28

**Stage 4 - C++/CUDA scaffolding:**
- csrc/kernels/ternary_gemm.cu: Baseline ternary GEMM kernel
- csrc/core/dispatch.cpp: ternary_linear_forward dispatch
- csrc/bindings/bittorch_bindings.cpp: Python bindings
- bittorch/ops/ternary_linear.py: Python wrapper functions

**Stage 5 - TernaryLinearCUDA module:**
- bittorch/nn/ternary_linear_cuda.py: CUDA-accelerated ternary linear layer
  - TernaryLinearCUDAFunction (torch.autograd.Function)
  - Forward uses CUDA kernel, backward uses PyTorch matmuls (STE)
- examples/xor_mlp.py: Added --cuda flag for CUDA backend
- tests/test_ternary_cuda.py: 32 tests for CUDA ops and module

Technical note: Weight gradients differ slightly between CUDA and Python
implementations due to scale path (see Journal/006_cuda_gradient_mismatch.md).
CUDA trains as well or better empirically.

---

### 4e948a8 - Stage 3: Tests & benchmark harness

**Date:** 2025-11-28

- Add 14 edge case tests for ternary quantization (extreme thresholds,
  small/large weights, single element tensors, etc.)
- Add quantize=False debug mode to TernaryLinear for comparison with nn.Linear
- Add 7 debug mode tests verifying exact match with nn.Linear when disabled
- Create benchmark/bench_ternary_linear.py with CLI flags and env vars
- Add Journal/ to .gitignore

Total: 58 tests passing. Benchmark establishes baseline: pure Python
TernaryLinear is ~6-10x slower than nn.Linear (expected, CUDA will fix).

---

### 5d693d6 - Stage 2: XOR MLP training baseline

**Date:** 2025-11-28

Proves that ternary quantization + STE + optimizer can converge:

- examples/xor_mlp.py:
  - 2→4→1 MLP using TernaryLinear
  - BCE loss on XOR truth table
  - Achieves 100% accuracy
  - Comparison with nn.Linear baseline

- tests/test_xor_mlp.py:
  - 5 regression tests for convergence
  - Verifies weights remain ternary
  - Checks gradients flow correctly

---

### ea10153 - Stage 1: Ternary quantization core in pure Python

**Date:** 2025-11-28

Implemented ternary quantization and TernaryLinear module:

- bittorch/quant/ternary.py:
  - ternary_quantize() for weight quantization to {-1, 0, +1}
  - ternary_quantize_ste() with Straight-Through Estimator
  - TernaryQuantConfig dataclass
  - dequantize_ternary() helper

- bittorch/nn/ternary_linear.py:
  - TernaryLinear drop-in replacement for nn.Linear
  - Full-precision master weights with STE gradient flow
  - Per-channel scaling by default

- 31 unit tests passing for quantization and gradient flow

---

### d19b5f7 - Initial commit: BitTorch v0.0.1 foundation (Stage 0)

**Date:** 2025-11-28

Set up the basic project structure for BitTorch, a high-performance
low-precision backend for PyTorch targeting ternary and INT4 weights.

- Python package structure (bittorch/) with ops, nn, quant, utils submodules
- C++/CUDA extension structure (csrc/) with dummy function
- Build system with setup.py using CUDAExtension
- pyproject.toml for modern packaging with uv
- Basic tests to verify extension loading
- uv.lock for reproducible builds

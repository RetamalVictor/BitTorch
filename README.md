# BitTorch

High-performance low-precision backend for PyTorch, targeting ternary (1.58-bit) weights.

## What is BitTorch?

BitTorch provides drop-in replacements for `nn.Linear` that use **ternary quantized weights** ({-1, 0, +1}) during forward pass while maintaining full-precision master weights for training. This enables:

- **~20x weight compression** (32-bit → 1.58-bit per weight)
- **Potential 2-4x speedups** with optimized kernels (multiplication becomes add/negate/skip)
- **Edge deployment** - fit larger models in constrained memory

Inspired by [BitNet b1.58](https://arxiv.org/abs/2402.17764) and related work on low-precision neural networks.

## Installation

Requires CUDA toolkit and PyTorch with CUDA support.

```bash
# Clone and install
git clone https://github.com/RetamalVictor/bittorch.git
cd bittorch

# Using uv (recommended)
uv sync
uv build

# Or using pip
pip install -e .
```

## Quick Start

```python
import torch
from bittorch.nn import TernaryLinear

# Drop-in replacement for nn.Linear
layer = TernaryLinear(64, 32)

# Forward pass uses ternary weights
x = torch.randn(8, 64)
y = layer(x)

# On CUDA: automatically uses optimized kernel
layer = TernaryLinear(64, 32).cuda()
x = torch.randn(8, 64, device='cuda')
y = layer(x)  # Uses CUDA kernel automatically

# Inspect quantized weights
w_tern, scale = layer.get_quantized_weight()
print(f"Unique values: {torch.unique(w_tern)}")  # tensor([-1., 0., 1.])
```

## Examples

### XOR MLP

```bash
# CPU (pure Python)
uv run python examples/xor_mlp.py

# GPU (CUDA kernel)
uv run python examples/xor_mlp.py --cuda
```

### MNIST MLP

```bash
# Compare ternary vs FP32
uv run python examples/mnist_mlp_ternary.py --compare --epochs 5
```

## Benchmark Results

### Training Accuracy (MNIST MLP)

784 → 256 → 128 → 10, 5 epochs:

| Model | Test Accuracy |
|-------|---------------|
| FP32 nn.Linear | 97.3% |
| TernaryLinear (CUDA) | 94.3% |

The ~3% accuracy gap is expected for 1.58-bit quantization.

### Forward Pass Performance

GPU: NVIDIA RTX A2000 8GB Laptop GPU

| Shape (B,K,N) | nn.Linear FP32 | nn.Linear FP16 | TernaryLinearCUDA | Ratio vs FP32 |
|---------------|----------------|----------------|-------------------|---------------|
| (32, 256, 256) | 0.03 ms | 0.03 ms | 0.18 ms | 5.1x slower |
| (64, 1024, 4096) | 0.28 ms | 0.10 ms | 5.4 ms | 19x slower |
| (16, 4096, 4096) | 0.66 ms | 0.29 ms | 12.8 ms | 19x slower |

**v0.1.3 Update:** Tiled kernel with shared memory provides 1.4-1.9x speedup over baseline.
Still slower than cuBLAS, but further optimizations planned.

Run benchmarks yourself:
```bash
uv run python benchmark/bench_ternary_vs_linear.py --shapes all
uv run python benchmark/bench_ternary_kernel_only.py  # Raw kernel timing
```

## API Reference

### `bittorch.nn.TernaryLinear`

```python
TernaryLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    threshold_factor: float = 0.05,  # Controls sparsity
    per_channel: bool = True,        # Per-channel vs global scaling
    quantize: bool = True,           # Set False for debug mode
    use_cuda_kernel: bool = True,    # Auto-use CUDA when available
)
```

### `bittorch.nn.TernaryLinearCUDA`

Explicit CUDA-only module (raises error on CPU input):

```python
from bittorch.nn import TernaryLinearCUDA

layer = TernaryLinearCUDA(64, 32).cuda()
```

## How It Works

1. **Master weights** stored in FP32 for optimizer updates
2. **Ternary quantization** on forward: `w_tern = sign(w) if |w| > threshold else 0`
3. **Per-channel scaling**: `scale = max(|w|)` per output channel
4. **Straight-Through Estimator (STE)**: gradients flow through quantization as if identity

The effective forward computation:
```
y = x @ (w_tern * scale).T + bias
```

## Project Status

**v0.1.3** - Current release:
- [x] Ternary quantization with STE
- [x] CUDA forward kernel (tiled, shared memory)
- [x] Training support (PyTorch backward)
- [x] Gradient consistency between Python and CUDA
- [x] XOR and MNIST examples
- [x] 120 tests passing

Coming in future versions:
- Extended examples (character LM) - v0.1.4
- Further kernel optimizations (vectorized loads)
- INT4 quantization path

## License

MIT

## Citation

```bibtex
@software{bittorch2024,
  author = {Victor Retamal},
  title = {BitTorch: High-performance low-precision backend for PyTorch},
  year = {2024},
  url = {https://github.com/RetamalVictor/bittorch}
}
```

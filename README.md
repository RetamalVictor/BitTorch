# BitTorch

High-performance ternary (1.58-bit) backend for PyTorch.

## What It Does

Drop-in `nn.Linear` replacements with **ternary quantized weights** ({-1, 0, +1}). Master weights stay in FP32 for training; quantization happens on forward pass via Straight-Through Estimator.

- ~20x weight compression (32-bit → 1.58-bit)
- CUDA kernel for GPU inference
- Works with standard PyTorch training loops

Inspired by [BitNet b1.58](https://arxiv.org/abs/2402.17764).

## Installation

```bash
git clone https://github.com/RetamalVictor/bittorch.git
cd bittorch
uv sync && uv build
```

## Quick Start

```python
import torch
from bittorch.nn import TernaryLinear

# Drop-in replacement for nn.Linear
layer = TernaryLinear(512, 256).cuda()
x = torch.randn(32, 512, device='cuda')
y = layer(x)

# Check quantized weights
w_tern, scale = layer.get_quantized_weight()
print(torch.unique(w_tern))  # tensor([-1., 0., 1.])
```

## Examples

```bash
# XOR MLP
uv run python examples/xor_mlp.py --cuda

# MNIST (compare ternary vs FP32)
uv run python examples/mnist_mlp_ternary.py --compare --epochs 5

# Character-level LM
uv run python examples/tiny_char_lm_ternary.py --cuda --epochs 10
```

## Benchmarks

```bash
uv run python benchmark/bench_ternary_vs_linear.py --shapes all
```

Current status: CUDA kernel is functional but slower than cuBLAS. Optimizations ongoing.

## API

```python
TernaryLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    threshold_factor: float = 0.05,
    backend: str = "auto",  # "auto", "cuda", "python"
)
```

## License

MIT

## Citation

```bibtex
@software{bittorch2025,
  author = {Victor Retamal},
  title = {BitTorch: Ternary quantization backend for PyTorch},
  year = {2025},
  url = {https://github.com/RetamalVictor/bittorch}
}
```

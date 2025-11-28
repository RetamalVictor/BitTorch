# BitTorch Capabilities Guide

This document explains what you can do with BitTorch v0.1.5.

## What is BitTorch?

BitTorch is a **high-performance low-precision backend for PyTorch** that provides drop-in replacements for `nn.Linear` using **ternary quantized weights** ({-1, 0, +1}).

### Key Benefits

1. **~20x weight compression**: 32-bit FP32 → 1.58-bit ternary
2. **Training support**: Full backpropagation via Straight-Through Estimator (STE)
3. **CUDA acceleration**: Optimized CUDA kernel with shared memory tiling
4. **Drop-in replacement**: Same API as `nn.Linear`

---

## What You Can Build

### 1. Compressed MLPs

Replace any `nn.Linear` with `TernaryLinear` for immediate compression:

```python
import torch
from bittorch.nn import TernaryLinear

# Standard MLP with ternary weights
class CompressedMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, hidden_dim)
        self.fc3 = TernaryLinear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Use it exactly like a normal model
model = CompressedMLP(784, 256, 10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training works normally
for x, y in dataloader:
    out = model(x)
    loss = criterion(out, y)
    loss.backward()  # Gradients flow through ternary weights!
    optimizer.step()
```

### 2. Image Classification (MNIST, CIFAR)

```bash
# Run the included MNIST example
uv run python examples/mnist_mlp_ternary.py --compare --epochs 10

# Expected results:
# - FP32 accuracy: ~97%
# - Ternary accuracy: ~94%
# - ~3% accuracy gap, but 20x smaller weights!
```

### 3. Character-Level Language Models

```bash
# Train on inline Shakespeare sample
uv run python examples/tiny_char_lm_ternary.py --compare --cuda

# Or download TinyShakespeare for larger training
uv run python examples/tiny_char_lm_ternary.py --download --compare --cuda --epochs 20
```

### 4. Edge Deployment

Ternary weights mean:
- Multiplication becomes add/negate/skip
- Weights fit in 2 bits (or less with encoding)
- Ideal for resource-constrained devices

```python
# Get quantized weights for deployment
layer = TernaryLinear(256, 128)
w_tern, scale = layer.get_quantized_weight()

# w_tern contains only {-1, 0, +1}
print(torch.unique(w_tern))  # tensor([-1.,  0.,  1.])

# Save for embedded deployment
torch.save({
    'w_tern': w_tern.to(torch.int8),  # 1 byte per weight
    'scale': scale,
    'bias': layer.bias,
}, 'compressed_layer.pt')
```

---

## API Reference

### TernaryLinear

```python
TernaryLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    threshold_factor: float = 0.05,  # Controls sparsity
    per_channel: bool = True,        # Per-channel vs global scaling
    quantize: bool = True,           # Set False for debug mode
    backend: str = "auto",           # "auto", "cuda", or "python"
)
```

**Backend options:**
- `"auto"` (default): Uses CUDA when available, otherwise Python
- `"cuda"`: Forces CUDA kernel (error if unavailable)
- `"python"`: Forces Python implementation (useful for debugging)

### Quantization Functions

```python
from bittorch.quant import ternary_quantize, ternary_quantize_ste

# Basic quantization
w_tern, scale = ternary_quantize(weight)

# With STE for training
w_tern_ste, scale = ternary_quantize_ste(weight)
```

### Quantizer Classes (New in v0.1.5)

```python
from bittorch.quant import TernaryQuantConfig, TernaryQuantizer

# Object-oriented quantization
config = TernaryQuantConfig(threshold_factor=0.1)
quantizer = TernaryQuantizer(config)

w_tern, scale = quantizer.quantize(weight)
w_recon = quantizer.dequantize(w_tern, scale)
```

---

## Performance

### Accuracy vs FP32

| Task | FP32 | Ternary | Gap |
|------|------|---------|-----|
| MNIST MLP | 97.3% | 94.3% | -3.0% |
| Char LM (PPL) | 1.00 | 1.33 | +33% |

### CUDA Kernel Performance

GPU: NVIDIA RTX A2000 8GB

| Shape (B,K,N) | nn.Linear FP32 | TernaryLinearCUDA | Ratio |
|---------------|----------------|-------------------|-------|
| (32, 256, 256) | 0.03 ms | 0.18 ms | 5x slower |
| (64, 1024, 4096) | 0.28 ms | 5.4 ms | 19x slower |

**Note:** The kernel is currently slower than cuBLAS but provides compression benefits. Future versions will close this gap with:
- Vectorized loads (`float4`)
- Ternary-specific optimizations (bit packing)
- Warp-level primitives

---

## What's NOT Supported Yet

### INT4 Quantization
The framework is ready (`Int4QuantConfig` placeholder exists), but INT4 is not implemented yet.

```python
from bittorch.quant import Int4QuantConfig

# This exists but raises NotImplementedError
config = Int4QuantConfig(group_size=128)
```

### Custom Backward Kernels
Forward pass uses CUDA, backward uses PyTorch matmuls. Full CUDA backward coming in future versions.

### Transformer Blocks
No attention or layer norm implementations yet. Use for feed-forward layers only.

---

## Development & Extension

### Adding Custom Quantization

Extend the base classes:

```python
from bittorch.quant.base import QuantConfig, Quantizer, QuantType

@dataclass
class MyQuantConfig(QuantConfig):
    my_param: float = 0.5

class MyQuantizer(Quantizer):
    def quantize(self, weight):
        # Your implementation
        pass
```

### Running Tests

```bash
# Fast tests (excludes slow big-shape tests)
uv run pytest tests/ -m 'not slow'

# All tests including slow
uv run pytest tests/
```

### Running Benchmarks

```bash
# Module-level benchmarks
uv run python benchmark/bench_ternary_vs_linear.py --shapes all

# Raw kernel benchmarks
uv run python benchmark/bench_ternary_kernel_only.py
```

---

## Next Steps

After getting familiar with BitTorch:

1. **Try the examples**: XOR, MNIST, character LM
2. **Integrate into your model**: Replace `nn.Linear` with `TernaryLinear`
3. **Benchmark your use case**: Measure accuracy-compression tradeoff
4. **Contribute**: See `CONTRIBUTING.md` for guidelines

---

## Resources

- [README.md](../README.md) - Quick start guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
- [docs/kernel_development.md](kernel_development.md) - CUDA kernel details
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [Journal/](../Journal/) - Implementation notes and decisions

# Contributing to BitTorch

This guide covers how to set up development, run tests, and contribute to BitTorch.

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA Toolkit 11.x or 12.x
- PyTorch 2.0+ with CUDA support
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/RetamalVictor/bittorch.git
cd bittorch

# Install with uv (recommended)
uv sync
uv sync --extra dev  # Include dev dependencies

# Or with pip
pip install -e ".[dev]"
```

### Building the CUDA Extension

The CUDA extension is built automatically on install, but you can rebuild manually:

```bash
# With uv
uv build

# Or with pip
pip install -e . --no-build-isolation
```

To verify the extension is loaded:

```python
import bittorch
print(bittorch.has_cuda_ext())  # Should print True
```

## Running Tests

```bash
# Run all tests (excluding slow tests)
uv run pytest tests/ -v -m 'not slow'

# Run all tests including slow tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_ternary_linear.py -v

# Run with coverage
uv run pytest tests/ --cov=bittorch --cov-report=html
```

### Test Markers

- `@pytest.mark.slow`: Long-running tests (big shapes, training loops)
- `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`: CUDA-only tests

## Running Benchmarks

### Full Module Benchmarks

```bash
# Compare TernaryLinear vs nn.Linear (FP32, FP16)
uv run python benchmark/bench_ternary_vs_linear.py --shapes all

# Specific shape
uv run python benchmark/bench_ternary_vs_linear.py --shapes 64,1024,4096
```

### Raw Kernel Benchmarks

```bash
# Compare baseline vs tiled vs cuBLAS
uv run python benchmark/bench_ternary_kernel_only.py
```

### Environment Variables

- `BITTORCH_KERNEL=baseline|tiled`: Force specific kernel variant
- `BITTORCH_DEVICE=cuda|cpu`: Override device selection

## Code Structure

```
bittorch/
├── __init__.py         # Package entry, version, has_cuda_ext()
├── nn/                 # Neural network modules
│   ├── ternary_linear.py      # TernaryLinear (main user API)
│   └── ternary_linear_cuda.py # CUDA autograd function
├── ops/                # Functional operations
│   └── ternary_linear.py      # ternary_linear() function
├── quant/              # Quantization utilities
│   ├── base.py         # Base QuantConfig, Quantizer classes
│   ├── ternary.py      # TernaryQuantConfig, ternary_quantize()
│   └── int4.py         # INT4 placeholder (future)
└── utils/              # Utilities

csrc/
├── core/
│   └── dispatch.cpp    # Kernel dispatch logic
├── kernels/
│   └── ternary_gemm.cu # CUDA kernels (baseline + tiled)
└── bindings/
    └── bittorch_bindings.cpp  # PyBind11 bindings

tests/                  # Test suite
benchmark/              # Performance benchmarks
examples/               # Usage examples
```

## Adding a New Kernel

### 1. Write the Kernel

Add your kernel to `csrc/kernels/`. Follow the existing pattern in `ternary_gemm.cu`:

```cpp
// csrc/kernels/my_kernel.cu

__global__ void my_kernel(...) {
    // Implementation
}

torch::Tensor my_kernel_forward(
    torch::Tensor x,
    torch::Tensor weight,
    // ...
) {
    // Launch kernel
}
```

### 2. Add Bindings

Update `csrc/bindings/bittorch_bindings.cpp`:

```cpp
m.def("my_kernel_forward", &my_kernel_forward, "My kernel forward");
```

### 3. Update setup.py

If adding new files, update `setup.py`:

```python
ext_modules=[
    CUDAExtension(
        name='bittorch._C',
        sources=[
            # ... existing sources ...
            'csrc/kernels/my_kernel.cu',
        ],
    )
],
```

### 4. Add Python Wrapper

Create `bittorch/ops/my_kernel.py` with the functional API.

### 5. Add Tests

Create `tests/test_my_kernel.py` with:
- Correctness tests (compare against reference implementation)
- Gradient tests (if applicable)
- Shape coverage tests

### 6. Add Benchmarks

Create `benchmark/bench_my_kernel.py` following existing patterns.

## Kernel Development Tips

### Tiling Strategy

Our tiled kernel uses 32x32x32 tiles:

```
TILE_M = 32  # Batch dimension
TILE_N = 32  # Output features
TILE_K = 32  # Reduction (input features)
```

Each thread block computes a 32x32 output tile by iterating over K-tiles.

### Shared Memory Usage

```cpp
__shared__ float X_tile[TILE_M][TILE_K];
__shared__ int8_t W_tile[TILE_N][TILE_K];
```

Total: 32*32*4 + 32*32*1 = 5120 bytes per block.

### Performance Pitfalls

1. **Global memory access patterns**: Ensure coalesced reads
2. **Bank conflicts**: Consider shared memory padding
3. **Occupancy**: Check with `--ptxas-options=-v`
4. **Boundary conditions**: Handle non-divisible shapes

### Debugging

```bash
# Force baseline kernel for comparison
BITTORCH_KERNEL=baseline uv run pytest tests/test_kernel_regression.py -v

# Check CUDA errors
CUDA_LAUNCH_BLOCKING=1 uv run python your_script.py
```

## Quantization Development

### Adding a New Quantization Scheme

1. Create config class inheriting from `QuantConfig`:

```python
# bittorch/quant/my_quant.py
from .base import QuantConfig, Quantizer, QuantType

@dataclass
class MyQuantConfig(QuantConfig):
    my_param: float = 0.5
```

2. Create quantizer class:

```python
class MyQuantizer(Quantizer):
    @property
    def quant_type(self) -> QuantType:
        return QuantType.MY_QUANT  # Add to enum

    def quantize(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        # Implementation
        pass
```

3. Create functional API:

```python
def my_quantize(weight, ...):
    # Implementation
    pass
```

4. Export in `__init__.py`

## Pull Request Guidelines

1. **One feature per PR**: Keep changes focused
2. **Tests required**: Add tests for new functionality
3. **Benchmarks for performance**: Include numbers if claiming speedups
4. **Documentation**: Update README/docstrings as needed

### PR Checklist

- [ ] All tests pass (`uv run pytest tests/ -m 'not slow'`)
- [ ] No new warnings
- [ ] Code follows existing style
- [ ] Docstrings for public APIs
- [ ] Benchmark data if performance-related

## Questions?

Open an issue at https://github.com/RetamalVictor/bittorch/issues

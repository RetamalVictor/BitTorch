# Kernel Development Guide

This document covers the architecture and implementation details of BitTorch's CUDA kernels.

## Overview

BitTorch currently implements a **ternary GEMM kernel** that computes:

```
Y = X @ (W_tern * scale)^T + bias
```

Where:
- `X`: Input activations `[B, K]` in FP32
- `W_tern`: Ternary weights `[N, K]` in {-1, 0, +1} (stored as int8)
- `scale`: Per-channel scaling factors `[N]`
- `Y`: Output `[B, N]` in FP32

## Kernel Variants

### 1. Baseline Kernel

Location: `csrc/kernels/ternary_gemm.cu` - `ternary_gemm_baseline_kernel`

Simple implementation where each thread computes one output element:

```cpp
// Thread (b, n) computes Y[b, n]
float acc = 0.0f;
for (int k = 0; k < K; k++) {
    int8_t w = W[n * K + k];
    acc += X[b * K + k] * (float)w;
}
Y[b * N + n] = acc * scale[n] + bias[n];
```

**Characteristics:**
- Simple and correct
- Poor memory reuse (each thread reads K elements from X)
- Used for small shapes or debugging

### 2. Tiled Kernel

Location: `csrc/kernels/ternary_gemm.cu` - `ternary_gemm_tiled_kernel`

Optimized implementation using shared memory tiling:

```cpp
constexpr int TILE_M = 32;  // Batch tile
constexpr int TILE_N = 32;  // Output tile
constexpr int TILE_K = 32;  // Reduction tile

__shared__ float X_tile[TILE_M][TILE_K];
__shared__ int8_t W_tile[TILE_N][TILE_K];
```

**Algorithm:**

1. Thread block computes a 32x32 output tile
2. For each K-tile:
   - Cooperatively load X and W tiles into shared memory
   - `__syncthreads()`
   - Each thread accumulates partial dot product from shared memory
3. Write results to global memory

**Characteristics:**
- 1.4-1.9x faster than baseline for medium/large shapes
- Better memory reuse (each element loaded once per tile)
- Still slower than cuBLAS (6-12x)

## Memory Layout

### Input Tensor X
- Shape: `[B, K]` (batch, input features)
- Memory: Row-major, contiguous
- Access pattern: Each thread block reads rows cooperatively

### Weight Tensor W_tern
- Shape: `[N, K]` (output features, input features)
- Memory: Row-major, stored as int8 values {-1, 0, +1}
- Access pattern: Transposed relative to cuBLAS convention

### Scale Vector
- Shape: `[N]`
- Applied after dot product: `Y[b,n] = dot * scale[n]`

## Dispatch Logic

Located in `csrc/core/dispatch.cpp`:

```cpp
bool use_baseline_kernel(int B, int N, int K) {
    // Check environment variable override
    const char* env = std::getenv("BITTORCH_KERNEL");
    if (env != nullptr) {
        if (std::string(env) == "baseline") return true;
        if (std::string(env) == "tiled") return false;
    }

    // Use baseline for tiny shapes (overhead dominates)
    if (B * N < 256 || K < 32) return true;

    return false;  // Default: tiled kernel
}
```

## Performance Analysis

### Current Status (v0.1.4)

GPU: NVIDIA RTX A2000 8GB Laptop GPU

| Shape (B,K,N) | Baseline | Tiled | Speedup | cuBLAS | vs cuBLAS |
|---------------|----------|-------|---------|--------|-----------|
| (32,256,256) | 97.7 us | 101.1 us | 0.97x | 27.8 us | 3.6x slower |
| (64,1024,4096) | 5927 us | 3159 us | 1.88x | 280 us | 11.3x slower |
| (32,4096,11008) | 31725 us | 18138 us | 1.75x | 1802 us | 10.1x slower |

### Why We're Slower Than cuBLAS

1. **No vectorized loads**: cuBLAS uses `float4` (128-bit) loads
2. **No warp-level primitives**: cuBLAS uses `__shfl_sync` for reductions
3. **Suboptimal tile sizes**: Not tuned per GPU architecture
4. **No ternary-specific tricks**: We still multiply by {-1,0,+1}

## Optimization Opportunities

### Short-term (Next Version)

1. **Vectorized loads**: Use `float4` for 4x memory bandwidth
   ```cpp
   float4 x_vec = *reinterpret_cast<float4*>(&X[offset]);
   ```

2. **Warp-level reduction**: Use `__shfl_down_sync` for faster accumulation

3. **Double buffering**: Overlap compute with memory loads

### Medium-term

1. **Ternary-specific encoding**: Pack 4 ternary values per byte (2 bits each)
   - Reduces memory bandwidth by 4x for weights
   - Requires custom unpack logic

2. **Avoid multiplication**: Ternary multiply is add/negate/skip
   ```cpp
   if (w == 1) acc += x;
   else if (w == -1) acc -= x;
   // else: w == 0, skip
   ```

3. **Tensor Core exploration**: Mixed-precision with INT8 Tensor Cores

### Long-term

1. **Autotuning**: Select tile sizes based on GPU architecture
2. **Sparse ternary**: Skip zero weights entirely
3. **Fused operations**: Combine quantization + GEMM in one kernel

## Adding a New Kernel

### Step 1: Write the Kernel

Create or modify `csrc/kernels/your_kernel.cu`:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void your_kernel(
    const float* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int B, int N, int K
) {
    // Your implementation
}

torch::Tensor your_kernel_forward(
    torch::Tensor x,
    torch::Tensor w_tern,
    torch::Tensor scale,
    torch::Tensor bias
) {
    // Shape checks
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(w_tern.dtype() == torch::kInt8, "W_tern must be int8");

    int B = x.size(0);
    int K = x.size(1);
    int N = w_tern.size(0);

    auto y = torch::empty({B, N}, x.options());

    // Launch kernel
    dim3 threads(...);
    dim3 blocks(...);

    your_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w_tern.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        B, N, K
    );

    return y;
}
```

### Step 2: Add Bindings

In `csrc/bindings/bittorch_bindings.cpp`:

```cpp
m.def("your_kernel_forward", &your_kernel_forward, "Your kernel forward");
```

### Step 3: Rebuild

```bash
uv build
```

### Step 4: Test

Create `tests/test_your_kernel.py`:

```python
def test_your_kernel_matches_reference():
    # Compare against Python reference
    pass

def test_your_kernel_gradients():
    # If supporting backward
    pass
```

## Profiling

### Basic Timing

```python
import torch
import bittorch._C as _C

x = torch.randn(64, 1024, device='cuda')
w = torch.randint(-1, 2, (4096, 1024), dtype=torch.int8, device='cuda')
scale = torch.ones(4096, device='cuda')
bias = torch.zeros(4096, device='cuda')

# Warmup
for _ in range(10):
    _C.ternary_linear_forward(x, w, scale, bias)
torch.cuda.synchronize()

# Time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    _C.ternary_linear_forward(x, w, scale, bias)
end.record()
torch.cuda.synchronize()

print(f"Average: {start.elapsed_time(end) / 100:.3f} ms")
```

### Nsight Compute

```bash
ncu --set full python your_script.py
```

Key metrics to watch:
- **Memory throughput**: Are we hitting DRAM bandwidth?
- **Compute throughput**: SM utilization
- **Occupancy**: Active warps per SM

## Common Issues

### 1. CUDA Errors

```bash
# Enable synchronous error checking
CUDA_LAUNCH_BLOCKING=1 python your_script.py
```

### 2. Wrong Results

```python
# Force baseline kernel for comparison
import os
os.environ["BITTORCH_KERNEL"] = "baseline"

# Compare
y_baseline = kernel_forward(...)

os.environ["BITTORCH_KERNEL"] = "tiled"
y_tiled = kernel_forward(...)

assert torch.allclose(y_baseline, y_tiled, atol=1e-5)
```

### 3. Slow Performance

Check:
1. Are inputs contiguous? `.contiguous()` if needed
2. Are shapes divisible by tile sizes?
3. Is the GPU warm? Run warmup iterations

## References

- [CUTLASS GEMM Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764)
